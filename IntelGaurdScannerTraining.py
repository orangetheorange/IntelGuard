import pandas as pd
import torch
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import re
import signal
import evaluate


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
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        src = self.embedding(input_ids)
        tgt = self.embedding(decoder_input_ids)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )

        output = self.transformer(
            src=src,
            tgt=tgt,
            src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        return self.fc_out(output)

    def generate(self, input_ids, attention_mask=None, max_length=50):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize with start token
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * tokenizer.bos_token_id

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )

            # Get last predicted token
            next_token_logits = outputs[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Append to sequence
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1
            )

            # Stop if all sequences have EOS
            if (decoder_input_ids == self.eos_token_id).any(dim=-1).all():
                break

        return decoder_input_ids


def raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, raise_keyboard_interrupt)


# Modified custom_loss function
def custom_loss(predictions, labels):
    # predictions shape: [batch_size, seq_len-1, vocab_size]
    # labels shape: [batch_size, seq_len-1]

    # Calculate standard cross-entropy loss
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(
        predictions.reshape(-1, predictions.size(-1)),
        labels.reshape(-1)
    )

    # Calculate penalty (modified for correct shapes)
    penalty = 0
    pred_indices = torch.argmax(predictions, dim=-1)  # [batch_size, seq_len-1]

    for i in range(pred_indices.size(0)):  # Loop through batch
        # Filter out padding tokens
        valid_mask = labels[i] != -100
        pred_tokens = pred_indices[i][valid_mask]
        true_tokens = labels[i][valid_mask]

        # Decode tokens to strings
        pred_str = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        true_str = tokenizer.decode(true_tokens, skip_special_tokens=True)

        # Penalty logic
        flags = re.findall(r'-\w+', pred_str)
        if len(flags) > len(set(flags)):
            penalty += 0.5
        if re.search(r'(-\w+)(?:\s+\1)+', pred_str):
            penalty += 0.5
        if pred_str == true_str:
            penalty -= 0.5

    return ce_loss + (penalty / pred_indices.size(0))  # Normalize by batch size

# Load your dataset
df = pd.read_csv("generated_dataset_50k.csv")
df["input_text"] = df.apply(lambda row: f"Target: {row['Target']}, Iden: {row['Iden']}, Stat: {row['Stat']}, Ports: {row['Open Ports']}, OS: {row['OS']}", axis=1)

inputs = df["input_text"].tolist()
targets = df["Custom Command"].tolist()

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
target_encodings = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return result


# Dataset class
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


# Dataset split
dataset = CommandDataset(input_encodings, target_encodings)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model instantiation
model = IntelGuardNet(32000, 256, 4, 4, 512)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)  # Reduced from 0.01
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10


def validation(model):
    model.eval()
    predictions, references = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Teacher forcing for loss calculation
            decoder_input_ids = batch["labels"][:, :-1]
            decoder_labels = batch["labels"][:, 1:]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_input_ids
            )

            loss = custom_loss(outputs, decoder_labels)
            total_loss += loss.item()

            # Generation
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=50
            )

            # Decoding
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    avg_val_loss = total_loss / len(val_dataloader)
    result = compute_metrics((predictions, references))
    print(f"Validation Loss: {avg_val_loss:.4f}, ROUGE: {result}")
    return avg_val_loss


# Training loop
print("Starting training...")

try:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Proper teacher-forcing setup
            decoder_input_ids = batch["labels"][:, :-1]  # Remove last token
            decoder_labels = batch["labels"][:, 1:]  # Remove first token

            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_input_ids
            )

            loss = custom_loss(outputs, decoder_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

        # Validation
        val_loss = validation(model)
        scheduler.step(val_loss)

except KeyboardInterrupt:
    userRequest = ""
    while userRequest.lower() not in ["y", "n"]:
        userRequest = input("\nTraining interrupted. Validate the model? (y/n): ")

    if userRequest.lower() == "y":
        validation(model)

    torch.save(model.state_dict(), 'model.pth')  # Save model state when interrupted

# Evaluation after training completion
print("Training finished. Starting evaluation...")
validation(model=model)


def generate_command(target, iden, stat, ports, os):
    input_text = f"Target: {target}, Iden: {iden}, Stat: {stat}, Ports: {ports}, OS: {os}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=50)
    command = tokenizer.decode(output[0], skip_special_tokens=True)
    return command
