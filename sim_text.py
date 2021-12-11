import numpy as np
from sentence_transformers import SentenceTransformer

def cosine(u, v):
    return float(np.dot(u, v)) / float((np.linalg.norm(u) * np.linalg.norm(v)))

def get_similarity_model():
	sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
	return sentence_model