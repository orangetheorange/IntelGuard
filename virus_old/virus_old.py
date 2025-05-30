import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from time import time
from torch.amp import GradScaler, autocast
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from torch.optim.lr_scheduler import OneCycleLR

# Hyperparameters
BATCH_SIZE = 256
INITIAL_LR = 0.01
MAX_LR = 0.1
EPOCHS = 19
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.05
HIDDEN_SIZE = 768
NUM_HEADS = 3

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
csv = pd.read_csv("malware_suspiciousness_dataset.csv")
X = csv.drop(columns=["Suspiciousness Level"]).values
y_raw = csv["Suspiciousness Level"].values


# Enhanced feature engineering
def create_polynomial_features(X, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
    return poly.fit_transform(X)


X = create_polynomial_features(X)

# Feature selection
selector = SelectKBest(mutual_info_classif, k=min(100, X.shape[1]))
X = selector.fit_transform(X, y_raw)

# Target encoding
unique_levels = sorted(set(y_raw))
level_to_class = {level: i for i, level in enumerate(unique_levels)}
class_to_level = {i: level for level, i in level_to_class.items()}
y = np.array([level_to_class[val] for val in y_raw])
num_classes = len(level_to_class)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Feature scaling with multiple transformers
preprocessor = QuantileTransformer(output_distribution='normal')
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# Advanced data augmentation
def augment_features(X, y, multiplier=2):
    X_augmented = [X]
    y_augmented = [y]
    for _ in range(multiplier):
        # Mixup augmentation
        lam = np.random.beta(0.4, 0.4)
        idx = np.random.permutation(len(X))
        X_augmented.append(lam * X + (1 - lam) * X[idx])
        y_augmented.append(y)

        # Gaussian noise
        noise = np.random.normal(0, 0.03, size=X.shape)
        X_augmented.append(X + noise)
        y_augmented.append(y)
    return np.vstack(X_augmented), np.concatenate(y_augmented)


X_train, y_train = augment_features(X_train, y_train, multiplier=2)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Enhanced class balancing
class_counts = np.bincount(y_train)
class_weights = 1. / (class_counts + 1e-6)
samples_weights = class_weights[y_train]
sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True
)


# Multi-Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.qkv = weight_norm(nn.Linear(input_dim, input_dim * 3))
        self.proj = weight_norm(nn.Linear(input_dim, input_dim))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        return self.gamma * out.squeeze(1) + x.squeeze(1)


# Ultra-optimized model architecture
class FastMalwareNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Feature extractor
        self.features = nn.Sequential(
            weight_norm(nn.Linear(input_dim, HIDDEN_SIZE)),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            MultiHeadAttention(HIDDEN_SIZE, NUM_HEADS),

            weight_norm(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            MultiHeadAttention(HIDDEN_SIZE // 2, NUM_HEADS),

            weight_norm(nn.Linear(HIDDEN_SIZE // 2, HIDDEN_SIZE // 4)),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # Multiple classification heads with skip connections
        self.head1 = nn.Sequential(
            weight_norm(nn.Linear(HIDDEN_SIZE // 4, HIDDEN_SIZE // 8)),
            nn.GELU(),
            weight_norm(nn.Linear(HIDDEN_SIZE // 8, num_classes))
        )
        self.head2 = nn.Sequential(
            weight_norm(nn.Linear(HIDDEN_SIZE // 4, HIDDEN_SIZE // 8)),
            nn.GELU(),
            weight_norm(nn.Linear(HIDDEN_SIZE // 8, num_classes))
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_bn(x)
        features = self.features(x)

        # Ensemble predictions
        pred1 = self.head1(features)
        pred2 = self.head2(features)

        return (pred1 + pred2) / 2


# Initialize model
input_dim = X_train_tensor.shape[1]
model = FastMalwareNet(input_dim, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function with class weighting and smoothing
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)



# Mixed precision training
scaler = GradScaler('cuda')

# Data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, pin_memory=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2,
                         pin_memory=True)
# Optimizer with OneCycle learning rate
optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-5)
scheduler = OneCycleLR(optimizer, max_lr=MAX_LR,
                       steps_per_epoch=len(train_loader),
                       epochs=EPOCHS,
                       anneal_strategy='cos')
# Training loop
best_val_acc = 0
train_losses, val_losses, val_accuracies = [], [], []

print(f"Training on {device}")
start_time = time()

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast('cuda'):
            preds = model(xb)
            loss = loss_fn(preds, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast('cuda'):
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()

            _, predicted = torch.max(preds.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total * 100
    val_accuracies.append(accuracy)
    val_losses.append(avg_val_loss)

    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), '../best_model.pth')

    print(f"Epoch {epoch + 1}/{EPOCHS}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {accuracy:.2f}%, "
          f"Best Acc: {best_val_acc:.2f}%, "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")


print(f"\nTraining completed in {(time() - start_time) / 60:.2f} minutes")

# Load best model
model.load_state_dict(torch.load('../best_model.pth'))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        with autocast('cuda'):
            preds = model(xb)
        _, predicted = torch.max(preds.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# Evaluation
target_names = [str(class_to_level[i]) for i in range(num_classes)]
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
print("Normalized Confusion Matrix:")
print(cm / cm.sum(axis=1)[:, np.newaxis])

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy')
plt.axhline(y=90, color='r', linestyle='--', label='90% Target')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()