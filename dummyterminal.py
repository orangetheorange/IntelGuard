import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.optim as optim


# dummy terminal function: returns a simulated output for allowed commands
def dummy_terminal(command):
    allowed_commands = {
        "ls": "file1.txt file2.txt",
        "pwd": "/home/dummy",
        "echo hello": "hello",
        "attack -fast": "attack successful: enemy stunned",
        "attack -optimal": "attack successful: enemy defeated"
    }
    return allowed_commands.get(command, "error: command not recognized or not allowed")


# reward function: returns +1 for valid commands, -1 for invalid ones
def compute_reward(command):
    output = dummy_terminal(command)
    return 1.0 if "error" not in output else -1.0


# sampling function to generate a sequence token-by-token
def sample_sequence(model, input_ids, max_length, tokenizer, device):
    # start with the decoder start token
    generated_ids = [model.config.decoder_start_token_id]
    log_probs = []
    for _ in range(max_length):
        decoder_input_ids = torch.tensor([generated_ids], device=device)
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # take logits for the last token in the sequence
        logits = outputs.logits[0, -1, :]
        distribution = torch.distributions.Categorical(logits=logits)
        token = distribution.sample().item()
        log_prob = distribution.log_prob(torch.tensor(token, device=device))
        log_probs.append(log_prob)
        generated_ids.append(token)
        if token == tokenizer.eos_token_id:
            break
    return generated_ids, log_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # dummy dataset: list of input prompts that combine your columns
    inputs = [
        "col1: sample1 | col2: sampleA | col3: data1 | col4: info1 | col5: detail1",
        "col1: sample2 | col2: sampleB | col3: data2 | col4: info2 | col5: detail2",
        "col1: sample3 | col2: sampleC | col3: data3 | col4: info3 | col5: detail3"
    ]

    num_epochs = 10
    max_length = 20  # maximum tokens to generate per command

    for epoch in range(num_epochs):
        total_loss = 0.0
        print(f"epoch {epoch + 1}/{num_epochs}")
        for input_text in inputs:
            optimizer.zero_grad()
            # encode input text
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            # sample a sequence from the model and record log probabilities
            generated_ids, log_probs = sample_sequence(model, input_ids, max_length, tokenizer, device)
            # remove the initial decoder start token for decoding
            command_ids = generated_ids[1:]
            generated_command = tokenizer.decode(command_ids, skip_special_tokens=True).strip()

            # get the reward by running the generated command in our dummy terminal
            reward = compute_reward(generated_command)
            # policy gradient loss: we want to maximize expected reward, so we minimize -reward * log_prob
            loss = - reward * torch.sum(torch.stack(log_probs))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"  input: {input_text}")
            print(f"  generated command: {generated_command}")
            print(f"  reward: {reward}, loss: {loss.item()}")
        print(f"total loss: {total_loss}\n")


if __name__ == "__main__":
    main()