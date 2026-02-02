import os
import numpy as np
from PIL import Image
import struct

# ============================
# BASE PATHS (EDIT ONLY HERE)
# ============================
BASE_DIR = r"E:\Final_project\Recommendation_Systems"
RAW_IDX_DIR = os.path.join(BASE_DIR, "raw_idx")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")

# ============================
# Create output folders
# ============================
for split in ["train", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ============================
# Fashion-MNIST class labels
# ============================
LABELS = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle_boot"
}

# ============================
# IDX file readers
# ============================
def read_images(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

    with open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images


def read_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

    with open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# ============================
# Save images (ImageFolder)
# ============================
def save_images(images, labels, split):
    print(f"📂 Saving {split} images...")

    for idx, (img, label) in enumerate(zip(images, labels)):
        class_name = LABELS[int(label)]
        class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Convert to PIL, RGB, resize for CNN
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = img.resize((224, 224))

        img_path = os.path.join(class_dir, f"{idx}.png")
        img.save(img_path)

    print(f"✅ {split} images saved successfully")


# ============================
# MAIN EXECUTION
# ============================
if __name__ == "__main__":
    print("🔄 Converting Fashion-MNIST IDX files to ImageFolder format...")

    # Train set
    train_images = read_images(os.path.join(RAW_IDX_DIR, "train-images-idx3-ubyte"))
    train_labels = read_labels(os.path.join(RAW_IDX_DIR, "train-labels-idx1-ubyte"))
    save_images(train_images, train_labels, "train")

    # Test set
    test_images = read_images(os.path.join(RAW_IDX_DIR, "t10k-images-idx3-ubyte"))
    test_labels = read_labels(os.path.join(RAW_IDX_DIR, "t10k-labels-idx1-ubyte"))
    save_images(test_images, test_labels, "test")

    print("🎉 Fashion-MNIST conversion COMPLETED successfully!")
