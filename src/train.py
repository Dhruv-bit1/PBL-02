import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

from src.dataset import VideoDataset
from src.model import FightDetector


# ==============================
# Device Setup
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==============================
# Transforms
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# Dataset Paths (Mini Demo)
# ==============================
train_path = "data/mini-test"
val_path = "data/mini-test"

train_dataset = VideoDataset(train_path, transform=transform)
val_dataset = VideoDataset(val_path, transform=transform)

print("Classes:", train_dataset.classes)
print("Number of videos:", len(train_dataset))


# ==============================
# DataLoaders (Windows Safe)
# ==============================
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ==============================
# Model Setup (Transfer Learning)
# ==============================
model = FightDetector(num_classes=2).to(device)

# Freeze backbone (train only last layer)
# Freeze all layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze final layer ONLY
for param in model.backbone.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


# ==============================
# Training Loop
# ==============================
epochs = 2

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for videos, labels in tqdm(train_loader):
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"\nEpoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

    # ==============================
    # Validation
    # ==============================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")


# ==============================
# Save Model
# ==============================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fight_model.pth")

print("\nTraining complete. Model saved to models/fight_model.pth")
