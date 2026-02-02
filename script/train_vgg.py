import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import VGG16_Weights

# =========================
# MAIN (Windows Safe)
# =========================
def main():

    # =========================
    # Device
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Using device:", device)

    # =========================
    # Paths
    # =========================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "train")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================
    # Transforms
    # =========================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # Dataset
    # =========================
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # 🔥 FAST DEMO SUBSET
    subset_size = min(2000, len(full_dataset))
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]

    train_data = Subset(full_dataset, subset_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    print("📂 Classes:", full_dataset.classes)
    print("🧪 Training images used:", len(train_data))

    # =========================
    # Model (VGG16)
    # =========================
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)

    # 🔥 Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=len(full_dataset.classes)
    )

    model = model.to(device)

    # =========================
    # Loss & Optimizer
    # =========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # =========================
    # Training Loop
    # =========================
    EPOCHS = 1
    print("🚀 Training started...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # =========================
    # Save Model
    # =========================
    model_path = os.path.join(MODEL_DIR, "vgg16_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print("💾 Model saved at:", model_path)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
