import torch
from transformers import T5Tokenizer
from IntelGuardNet import IntelGuardNet

from pathlib import Path
import os

script_path = Path(__file__).resolve()

parent_dir = script_path.parent

os.chdir(parent_dir)

# 1. Load the tokenizer
print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
tokenizer.add_special_tokens({'additional_special_tokens': ['[CMD]', '[FLAG]', '[VAL]']})

# 2.1 recreate your model skeleton exactly as you did at training time
print("building model architecture…")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntelGuardNet(
    vocab_size=tokenizer.vocab_size + len(tokenizer.additional_special_tokens),
    d_model=256,
    nhead=4,
    num_layers=4,
    dim_feedforward=512,
    tokenizer=tokenizer
).to(device)

# 2.2 load only the weights you saved (final_model.pth is just a state_dict)
print("loading weights from final_model.pth…")
state_dict = torch.load(
    r"final_model.pth",
    map_location=device
)
model.load_state_dict(state_dict)
model.eval()


def generate_command(input_text, max_len=64):
    # tokenize encoder input
    enc = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    # start decoder with pad token
    decoder_ids = torch.full((1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        decoder_ids = torch.cat([decoder_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    # skip initial pad token
    gen_ids = decoder_ids[:, 1:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

def predict_command(raw):
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 5:
        raise ValueError("expected 5 comma-separated fields")
    target, iden, stat, ports, os_ = parts
    # rebuild string exactly as in training ("Ports:", not "Open Ports:")
    input_text = (
        f"Target: {target}, "
        f"Iden: {iden}, "
        f"Stat: {stat}, "
        f"Ports: {ports}, "
        f"OS: {os_}"
    )
    print("encoder input ids:", tokenizer(input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids)
    return generate_command(input_text)


# 4. Main application loop
if __name__ == "__main__":
    print("Welcome to IntelGuardNet Command Generator!")

    while True:
        print("\nEnter the scan information (or type 'exit' to quit):")
        raw_input = input("Enter scan line: ")  # one line

        # Split it by commas
        parts = [p.strip() for p in raw_input.split(",")]

        # Debug: Print the parts of the input
        print(f"Parts after splitting input: {parts}")

        if len(parts) != 5:
            print("Invalid input! Must be: target, identity, status, open_ports, os")
            continue

        target, identity, status, open_ports, os = parts

        # Create input text for the model
        input_text = f"Target: {target}, Iden: {identity}, Stat: {status}, Open Ports: {open_ports}, OS: {os}"

        # Debug: Print the final input text
        print(f"Input text for model: {input_text}")

        # Get the predicted command
        prediction = predict_command(input_text)

        # Output the prediction
        print(f"\n[Predicted Command]: {prediction}")
