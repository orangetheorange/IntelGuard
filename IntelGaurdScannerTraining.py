# 1. Import libraries
import pandas as pd
import torch
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import evaluate
from tqdm import tqdm
from IntelGuardNet import IntelGuardNet
from pathlib import Path
import os

script_path = Path(__file__).resolve()

parent_dir = script_path.parent

os.chdir(parent_dir)

# 2. Data Preparation with validation
print("Loading and preparing data...")
df = pd.read_csv("generated_dataset_100000_lines.csv")
df["input_text"] = df.apply(lambda
                                row: f"Target: {row['Target']}, Iden: {row['Iden']}, Stat: {row['Stat']}, Ports: {row['Open Ports']}, OS: {row['OS']}",
                            axis=1)

tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
tokenizer.add_special_tokens({'additional_special_tokens': ['[CMD]', '[FLAG]', '[VAL]']})

input_encodings = tokenizer(df["input_text"].tolist(), padding=True, truncation=True, max_length=128,
                            return_tensors="pt")
target_encodings = tokenizer(df["Custom Command"].tolist(), padding=True, truncation=True, max_length=128,
                             return_tensors="pt")


# 3. Dataset Class with better batching
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


# 4. Improved Training Setup
dataset = CommandDataset(input_encodings, target_encodings)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = IntelGuardNet(
    vocab_size=tokenizer.vocab_size + len(tokenizer.additional_special_tokens),
    d_model=256,
    nhead=4,
    num_layers=4,
    dim_feedforward=512,
    tokenizer=tokenizer
).to(device)

# 5. Enhanced Training Configuration
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    steps_per_epoch=len(train_dataloader),
    epochs=5,
    pct_start=0.1
)


def custom_loss(predictions, labels, tokenizer):
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


        # Reward: If the prediction matches the true string
        if pred_str == true_str:
            penalty -= 0.5

    # Return the total loss
    avg_penalty = penalty / batch_size  # Average penalty over the batch
    return ce_loss + avg_penalty

# Enhanced evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


def compute_metrics(predictions, labels, tokenizer):
    decoded_preds = tokenizer.batch_decode(torch.argmax(predictions, dim=-1), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Basic accuracy
    mask = labels != tokenizer.pad_token_id
    correct = (torch.argmax(predictions, dim=-1) == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    # ROUGE and BLEU scores
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_scores = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])

    return {
        "accuracy": accuracy.item(),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_scores["bleu"]
    }


def validation(model, dataloader, tokenizer):
    model.eval()
    total_loss = 0
    exact_matches = 0  # Added to track exact matches
    all_metrics = {
        "accuracy": 0,
        "rouge1": 0,
        "rouge2": 0,
        "rougeL": 0,
        "bleu": 0
    }

    # Create a single progress bar at the start
    val_bar = tqdm(dataloader, desc="Validating", colour='green',
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')

    with torch.no_grad():
        for batch in val_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["labels"][:, :-1]
            )
            
            loss = custom_loss(outputs, batch["labels"][:, 1:], tokenizer)
            total_loss += loss.item()

            # Calculate exact matches
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(batch["input_ids"])):
                pred_str = tokenizer.decode(preds[i], skip_special_tokens=True)
                true_str = tokenizer.decode(batch["labels"][i, 1:], skip_special_tokens=True)
                if pred_str == true_str:
                    exact_matches += 1

            metrics = compute_metrics(outputs, batch["labels"][:, 1:], tokenizer)
            for k in all_metrics:
                all_metrics[k] += metrics[k]

    # Close the progress bar when done
    val_bar.close()

    avg_loss = total_loss / len(dataloader)
    exact_match_rate = exact_matches / len(dataloader.dataset)  # Calculate exact match rate
    for k in all_metrics:
        all_metrics[k] /= len(dataloader)

    # Print samples
    print("Validation Samples:")
    test_batch = next(iter(dataloader))
    test_batch = {k: v.to(device) for k, v in test_batch.items()}
    sample_outputs = model(
        input_ids=test_batch["input_ids"][:3],
        attention_mask=test_batch["attention_mask"][:3],
        decoder_input_ids=test_batch["labels"][:3, :-1]
    )
    sample_preds = torch.argmax(sample_outputs, dim=-1)

    for i in range(3):
        print(f"Sample {i + 1}:")
        print("Input:", tokenizer.decode(test_batch["input_ids"][i], skip_special_tokens=True))
        print("Predicted:", tokenizer.decode(sample_preds[i], skip_special_tokens=True))
        print("Actual:", tokenizer.decode(test_batch["labels"][i, 1:], skip_special_tokens=True))

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Exact Match Rate: {exact_match_rate:.2%}")
    print("Metrics:", all_metrics)
    return avg_loss, all_metrics

# 6. Training Loop
best_val_loss = float('inf')
for epoch in range(1):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
#         print("batch at itter",  batch["input_ids"])
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["input_ids"][:, :-1]
        )

        loss = custom_loss(outputs, batch["labels"][:, 1:], tokenizer)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    avg_train_loss = epoch_loss / len(train_dataloader)

    # Validation
    val_loss, val_metrics = validation(model, val_dataloader, tokenizer)
    print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #torch.save(model.state_dict(), "best_model.pth")
        # print("Saved new best model")

    # Save checkpoint
    #torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': avg_train_loss,
    #     'val_loss': val_loss
    # }, f"checkpoint_epoch_{epoch + 1}.pth")

# 7. Final Evaluation and Model Saving
print("Training complete. Running final evaluation...")
final_val_loss, final_metrics = validation(model, val_dataloader, tokenizer)
print(f"Final Validation Loss: {final_val_loss:.4f}")
print("Final Metrics:", final_metrics)

# Save final model
torch.save(model.state_dict(), "test_model.pth")
#torch.save(model, "100k/model_full.pth")
print("Model saved successfully")
