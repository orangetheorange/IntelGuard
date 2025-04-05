import pandas as pd
import torch
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import evaluate
import re

# 1. Model Definition
class IntelGuardNet(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(IntelGuardNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True  # Ensure batch_first is True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        # Ensure input and target sequences have the same batch size
        src = self.embedding(input_ids)  # Shape: [batch_size, seq_len, d_model]
        tgt = self.embedding(decoder_input_ids)  # Shape: [batch_size, seq_len, d_model]

        # Transformer expects input shape [batch_size, seq_len, d_model]
        output = self.transformer(
            src,  # Shape: [batch_size, seq_len, d_model]
            tgt  # Shape: [batch_size, seq_len, d_model]
        )

        # The output shape is [batch_size, seq_len, d_model]
        return self.fc_out(output)  # Shape: [batch_size, seq_len, vocab_size]


# 2. Data Preparation
df = pd.read_csv("generated_dataset_50k.csv")
df["input_text"] = df.apply(lambda
                                row: f"Target: {row['Target']}, Iden: {row['Iden']}, Stat: {row['Stat']}, Ports: {row['Open Ports']}, OS: {row['OS']}",
                            axis=1)

tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
input_encodings = tokenizer(df["input_text"].tolist(), padding=True, truncation=True, return_tensors="pt")
target_encodings = tokenizer(df["Custom Command"].tolist(), padding=True, truncation=True, return_tensors="pt")


# 3. Dataset Class
class CommandDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_encodings["input_ids"][idx],
            "attention_mask": self.input_encodings["attention_mask"][idx],
            "labels": self.target_encodings["input_ids"][idx],
        }


# 4. Training Setup
dataset = CommandDataset(input_encodings, target_encodings)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntelGuardNet(tokenizer.vocab_size, 256, 4, 4, 512).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 5. Evaluation Functions
rouge = evaluate.load("rouge")

def custom_loss(predictions, labels, tokenizer):
    """
    Custom loss function with cross-entropy and additional penalties for certain conditions.

    Arguments:
    - predictions: The model's output, shape [batch_size, seq_len-1, vocab_size]
    - labels: The ground truth labels, shape [batch_size, seq_len-1]
    - tokenizer: The tokenizer used to decode token IDs to strings

    Returns:
    - Total loss (cross-entropy loss + penalties)
    """
    # Calculate standard cross-entropy loss
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1))

    # Calculate penalty (based on repeated tokens, flags, nmap-like commands, and IP addresses)
    penalty = 0
    batch_size = predictions.size(0)
    pred_indices = torch.argmax(predictions, dim=-1)  # Predicted tokens [batch_size, seq_len-1]

    # Get special tokens
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    # Loop through the batch to compute the penalties
    for i in range(batch_size):
        # Filter out padding tokens
        valid_mask = (labels[i] != -100) & (labels[i] != pad_token_id) & (labels[i] != eos_token_id)
        pred_tokens = pred_indices[i][valid_mask]
        true_tokens = labels[i][valid_mask]

        # Decode tokens to strings
        pred_str = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        true_str = tokenizer.decode(true_tokens, skip_special_tokens=True)

        # Penalty 1: Prediction doesn't start with "nmap" or IP (last part if split with space)
        if not (pred_str.startswith("nmap") or
               (len(pred_str.split()) > 0 and pred_str.split()[-1].replace('.', '').isdigit())):
            penalty += 0.5

        # Penalty 2: Single '-' not followed by another '-' or a letter
        # Find all '-' in the prediction
        for j in range(len(pred_str)-1):
            if pred_str[j] == '-' and not (pred_str[j+1] == '-' or pred_str[j+1].isalpha()):
                penalty += 0.3

        # Penalty 3: Repeating flags or tokens in prediction
        flags = re.findall(r'-\w+', pred_str)  # Find all flags in the prediction
        if len(flags) > len(set(flags)):  # If flags are repeated
            penalty += 0.5
        if re.search(r'(-\w+)(?:\s+\1)+', pred_str):  # If the same flag appears consecutively
            penalty += 0.5

        # Reward: If the prediction matches the true string
        if pred_str == true_str:
            penalty -= 0.5

    # Return the total loss
    avg_penalty = penalty / batch_size  # Average penalty over the batch
    return ce_loss + avg_penalty


def compute_metrics(predictions, labels):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)


def validation(model):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["labels"][:, :-1]
            )

            # Apply custom loss function
            loss = custom_loss(outputs, batch["labels"][:, 1:], tokenizer)  # Pass predictions, labels, and tokenizer

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"][:, 1:].cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels)
    avg_loss = total_loss / len(val_dataloader)

    print("\nValidation Samples:")
    for i in range(3):
        print(f"Input: {tokenizer.decode(val_dataset[i]['input_ids'], skip_special_tokens=True)}")
        print(f"Predicted: {tokenizer.decode(all_preds[i], skip_special_tokens=True)}")
        print(f"Actual: {tokenizer.decode(all_labels[i], skip_special_tokens=True)}\n")

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"ROUGE Scores: {metrics}")
    return avg_loss



# 6. Training Loop
for epoch in range(10):
    model.train()
    epoch_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["labels"][:, :-1]
        )

        loss = nn.CrossEntropyLoss()(outputs.reshape(-1, outputs.size(-1)),
                                      batch["labels"][:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss {loss.item():.4f}")

    print(f"\nEpoch {epoch + 1} Train Loss: {epoch_loss / len(train_dataloader):.4f}")
    val_loss = validation(model)

    # Save model checkpoint
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

# 7. Final Evaluation
print("\nTraining complete. Running final evaluation...")
final_val_loss = validation(model)
print(f"Final Validation Loss: {final_val_loss:.4f}")


# 8. Inference Function
def generate_command(model, target, iden, stat, ports, os):
    input_text = f"Target: {target}, Iden: {iden}, Stat: {stat}, Ports: {ports}, OS: {os}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Simple greedy decoding
    decoder_input = torch.tensor([[tokenizer.pad_token_id]]).to(device)
    for _ in range(50):  # max length
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input
        )
        next_token = torch.argmax(outputs[:, -1, :], dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(decoder_input[0], skip_special_tokens=True)
