import os
import random
import torch
import numpy as np
import pickle
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights


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
    FEATURE_DIR = os.path.join(BASE_DIR, "features")
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # =========================
    # Model (ResNet50)
    # =========================
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

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

    subset_size = min(2000, len(full_dataset))
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]

    dataset = Subset(full_dataset, subset_indices)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    print("🧪 Total images used for feature extraction:", len(dataset))

    # =========================
    # Feature Extraction
    # =========================
    features = []

    print("🚀 Feature extraction started...")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(loader)}")

    features = np.vstack(features)

    # =========================
    # Save features & paths
    # =========================
    features_path = os.path.join(FEATURE_DIR, "image_embeddings.npy")
    paths_path = os.path.join(FEATURE_DIR, "image_paths.pkl")

    np.save(features_path, features)

    subset_paths = [full_dataset.imgs[i] for i in subset_indices]
    with open(paths_path, "wb") as f:
        pickle.dump(subset_paths, f)

    print("✅ Feature extraction completed successfully!")
    print("💾 Saved embeddings to:", features_path)
    print("💾 Saved image paths to:", paths_path)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
