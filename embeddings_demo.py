from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Your "database"
documents = [
    "I love programming",
    "Coding is fun",
    "I enjoy cooking",
    "Football is a great sport",
    "Software development is interesting"
]

# Convert docs to embeddings
doc_embeddings = model.encode(documents)

# Ask a question
query = "I like coding"

query_embedding = model.encode(query)

# Compute similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]

# Get best match
best_idx = np.argmax(scores)

print("Query:", query)
print("Best match:", documents[best_idx])
print("Score:", scores[best_idx])