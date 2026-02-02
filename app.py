import os
import torch
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import ResNet50_Weights, VGG16_Weights
import torch.nn.functional as F

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Image Recommendation System",
    layout="wide"
)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")

EMBED_PATH = os.path.join(FEATURE_DIR, "image_embeddings.npy")
PATHS_PATH = os.path.join(FEATURE_DIR, "image_paths.pkl")
VGG_PATH = os.path.join(MODEL_DIR, "vgg16_classifier.pth")

# =====================================================
# SAFETY CHECKS
# =====================================================
if not (os.path.exists(EMBED_PATH) and os.path.exists(PATHS_PATH)):
    st.error("❌ Feature files not found. Run extract_features.py first.")
    st.stop()

if not os.path.exists(VGG_PATH):
    st.error("❌ VGG16 model not found. Run train_vgg.py first.")
    st.stop()

# =====================================================
# CLASS NAMES (Fashion-MNIST)
# =====================================================
CLASS_NAMES = [
    "Ankle_boot", "Bag", "Coat", "Dress", "Pullover",
    "Sandal", "Shirt", "Sneaker", "T-shirt", "Trouser"
]

# =====================================================
# LOAD FEATURES & PATHS (CACHED)
# =====================================================
@st.cache_resource
def load_features():
    features = np.load(EMBED_PATH)
    with open(PATHS_PATH, "rb") as f:
        paths = pickle.load(f)
    return features, paths

features, image_paths = load_features()

# =====================================================
# LOAD MODELS (CACHED)
# =====================================================
@st.cache_resource
def load_models():
    # ResNet50 for similarity
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval().to(device)

    # VGG16 for classification
    vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
    vgg.classifier[6] = torch.nn.Linear(4096, len(CLASS_NAMES))
    vgg.load_state_dict(torch.load(VGG_PATH, map_location=device))
    vgg.eval().to(device)

    return resnet, vgg

resnet, vgg = load_models()

# =====================================================
# TRANSFORMS
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# UI
# =====================================================
st.title("🖼️ Image Recommendation System (CNN Based)")
st.write("Upload an image to get its class and visually similar recommendations.")

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # =================================================
    # CLASS PREDICTION
    # =================================================
    with torch.no_grad():
        logits = vgg(img_tensor)
        probs = F.softmax(logits, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, class_idx].item()

    predicted_class = CLASS_NAMES[class_idx]

    st.subheader("📌 Predicted Class")
    st.success(f"{predicted_class}  (Confidence: {confidence*100:.2f}%)")

    # =================================================
    # FEATURE EXTRACTION
    # =================================================
    with torch.no_grad():
        query_feature = resnet(img_tensor).cpu().numpy()

    TOP_K = 5

    # =================================================
    # RECOMMENDATION LOGIC
    # =================================================
    if confidence >= 0.60:
        # Filter by predicted class
        filtered_indices = [
            i for i, (_, label) in enumerate(image_paths)
            if label == class_idx
        ]

        if len(filtered_indices) < TOP_K:
            filtered_indices = list(range(len(image_paths)))
            st.warning("⚠️ Few images in this class — using global similarity.")
        else:
            st.caption("🔒 Recommendations filtered by predicted class")

        filtered_features = features[filtered_indices]
        similarity = cosine_similarity(query_feature, filtered_features)[0]
        top_local = similarity.argsort()[-TOP_K:][::-1]
        top_indices = [filtered_indices[i] for i in top_local]

    else:
        # Global similarity fallback
        similarity = cosine_similarity(query_feature, features)[0]
        top_indices = similarity.argsort()[-TOP_K:][::-1]
        st.warning("⚠️ Low confidence — showing global similar images")

    # =================================================
    # DISPLAY RESULTS
    # =================================================
    st.subheader("🔍 Recommended Images")
    cols = st.columns(TOP_K)

    for col, idx in zip(cols, top_indices):
        img_path, label = image_paths[idx]
        col.image(img_path, use_column_width=True)
        col.caption(CLASS_NAMES[label])
