import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fix random seeds
torch.manual_seed(42)
np.random.seed(42)

# --- Load and preprocess data ---
print("Loading data...")
csv = pd.read_csv("malware_suspiciousness_dataset.csv")
X = csv.drop(columns=["Suspiciousness Level"]).values
y_raw = csv["Suspiciousness Level"].values

# Encode labels
levels = sorted(set(y_raw))
level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
y = np.array([level_to_idx[lvl] for lvl in y_raw])
num_classes = len(levels)

print("\nClass distribution in original data:")
print(pd.Series(y_raw).value_counts())

# Feature selection (simpler approach)
print("\nPerforming feature selection...")
selector = SelectKBest(mutual_info_classif, k=min(50, X.shape[1]))  # Reduced feature count
X_selected = selector.fit_transform(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42  # Slightly larger test set
)

# Scaling (using StandardScaler instead of QuantileTransformer)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Handle class imbalance
class_counts = np.bincount(y_train.numpy())
weights = 1. / (class_counts + 1e-6)
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# --- Model Definition ---
class MalwareClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # Initialize weights properly (outside the Sequential block)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)



# --- Training Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = MalwareClassifier(X_train.shape[1], num_classes).to(device)
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Data loaders
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    sampler=sampler,
    pin_memory=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=128,
    shuffle=False,
    pin_memory=True
)


# --- Training Loop ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return running_loss / len(loader), 100 * correct / total


print("\nStarting training...")
best_acc = 0
train_losses, val_losses, train_accs, val_accs = [], [], [], []
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

for epoch in range(100):  # Increased max epochs
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Print progress
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1:03d}: "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"LR: {lr:.2e}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_malware_model.pth")
        print(f"New best model saved with accuracy {best_acc:.2f}%")

    # Early stopping if no improvement for 15 epochs
    if epoch > 15 and best_acc not in val_accs[-15:]:
        print("Early stopping triggered")
        break

# --- Final Evaluation ---
print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load("best_malware_model.pth"))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[str(lvl) for lvl in levels], digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# --- Plot Training Curves ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title("Training and Validation Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
