import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

features = np.load("features/image_embeddings.npy")

def recommend(query_feature, top_k=5):
    similarity = cosine_similarity(query_feature.reshape(1, -1), features)
    indices = similarity.argsort()[0][-top_k:][::-1]
    return indices
