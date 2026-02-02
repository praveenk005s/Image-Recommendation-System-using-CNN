import os
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import Counter

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "train")

# =========================
# TRANSFORM (NO NORMALIZE FOR DISPLAY)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# LOAD DATASET
# =========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

class_names = dataset.classes
labels = [label for _, label in dataset.samples]

print("✅ Dataset Loaded Successfully")
print("Total Images:", len(dataset))
print("Classes:", class_names)

# =========================
# 1️⃣ CLASS DISTRIBUTION
# =========================
class_count = Counter(labels)

print("\n📊 Images per class:")
for idx, class_name in enumerate(class_names):
    print(f"{class_name}: {class_count[idx]}")

# Plot class distribution
plt.figure(figsize=(10, 5))
plt.bar(class_names, [class_count[i] for i in range(len(class_names))])
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# =========================
# 2️⃣ SAMPLE IMAGE VISUALIZATION
# =========================
def show_sample_images(dataset, classes, samples_per_class=2):
    plt.figure(figsize=(12, 8))
    idx = 1

    for class_idx, class_name in enumerate(classes):
        class_images = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_idx
        ]

        selected_images = random.sample(class_images, samples_per_class)

        for img_idx in selected_images:
            image, label = dataset[img_idx]
            image = image.permute(1, 2, 0)

            plt.subplot(len(classes), samples_per_class, idx)
            plt.imshow(image)
            plt.axis("off")

            if idx <= samples_per_class:
                plt.title(class_name)

            idx += 1

    plt.suptitle("Sample Images per Class", fontsize=16)
    plt.tight_layout()
    plt.show()

show_sample_images(dataset, class_names)

# =========================
# 3️⃣ IMAGE SIZE ANALYSIS
# =========================
widths = []
heights = []

for img_path, _ in dataset.samples[:500]:  # check first 500 images
    from PIL import Image
    img = Image.open(img_path)
    w, h = img.size
    widths.append(w)
    heights.append(h)

print("\n📐 Image Size Statistics:")
print("Width  -> Min:", min(widths), "Max:", max(widths))
print("Height -> Min:", min(heights), "Max:", max(heights))

# =========================
# 4️⃣ DATASET BALANCE CHECK
# =========================
balanced = all(
    class_count[i] == class_count[0]
    for i in range(len(class_names))
)

print("\n⚖️ Dataset Balance Check:")
if balanced:
    print("✅ Dataset is balanced")
else:
    print("⚠️ Dataset is NOT balanced")

print("\n🎯 EDA Completed Successfully!")
