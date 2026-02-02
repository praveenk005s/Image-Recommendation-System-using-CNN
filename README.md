🖼️ Fashion-MNIST Image Recommendation System using CNN

An end-to-end Deep Learning project that builds an image-based recommendation system using Convolutional Neural Networks (CNNs).
The system classifies fashion images and recommends visually similar items using deep feature embeddings.

📌 Project Overview

This project demonstrates how to:

Convert raw Fashion-MNIST IDX files into image folders

Perform Exploratory Data Analysis (EDA)

Train a CNN classifier (VGG16)

Extract deep feature embeddings (ResNet50)

Build an image similarity recommendation system

Deploy the model using Streamlit

🚀 Demo Features

✅ Upload any fashion image

✅ Predict fashion category with confidence score

✅ Recommend visually similar images

✅ Filter recommendations by predicted class

✅ Real-time inference using Streamlit

📂 Project Structure
Recommendation_Systems/
│
├── data/
│   └── raw/
│       ├── train/
│       └── test/
│
├── raw_idx/                  # Original IDX files
│
├── script/
│   ├── convert_fashion_mnist.py
│   ├── eda_fashion_mnist.py
│   ├── train_vgg.py
│   ├── extract_features.py
│   ├── data_loader.py
│   ├── recommend.py
│   
│
├── features/
│   ├── image_embeddings.npy
│   └── image_paths.pkl
│
├── models/
│   └── vgg16_classifier.pth
│
├── requirements.txt
└── README.md
└── app.py

🧠 Dataset

Fashion-MNIST

60,000 training images

10,000 test images

10 clothing categories

Classes

T-shirt, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, Ankle boot

🔍 Exploratory Data Analysis (EDA)

Performed:

Class distribution analysis

Sample image visualization

Image size inspection

Dataset balance verification

📊 Result: Dataset is balanced across all classes.

🏗 Model Architecture
🔹 Classification Model

VGG16 (Transfer Learning)

Frozen convolution layers

Custom classifier head

Softmax output (10 classes)

🔹 Feature Extraction Model

ResNet50

Final classification layer removed

2048-dimensional feature embeddings

🔄 Recommendation Logic

Extract deep features for all dataset images

Extract features from uploaded image

Compute cosine similarity

Filter by predicted class (if confidence ≥ 60%)

Return Top-K similar images

🖥️ Streamlit Web App

Features:

Image upload

Class prediction + confidence

Similar image recommendation

Class-based filtering

Clean UI

Run app:

streamlit run script/app.py

📈 Skills Gained from This Project

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

CNN Architecture & Transfer Learning

Feature Engineering for Deep Learning

Image Similarity Search

Model Optimization

Model Deployment (Streamlit)

End-to-End ML Pipeline Design

🎯 Use Cases

Fashion recommendation systems

Visual search engines

E-commerce product similarity

Computer vision portfolios

Deep learning demonstrations

🧑‍💻 Author

Praveen Kumar
📍 India
📧 (praveenk005s@gmail.com)
🔗 (Linkedin :https://www.linkedin.com/in/praveenkumars021/)
