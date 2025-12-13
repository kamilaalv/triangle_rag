import faiss
import os
import uuid
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to chunk the text into smaller pieces with overlap
def chunk_text(text, size=1000, overlap=150):
    chunks = []
    for i in range(0, len(text), size - overlap):  # Apply overlap for smoother context retrieval
        chunks.append(text[i:i+size])
    return chunks


# Read all .txt files from out_txt directory
texts = []
for file in os.listdir("data/out_txt"):
    if file.endswith(".txt"):
        with open(f"data/out_txt/{file}", "r", encoding="utf-8") as f:
            texts.append(f.read())

# Chunk the text and create embeddings
chunks = []
for idx, text in enumerate(texts):
    chunked_texts = chunk_text(text)
    for chunk in chunked_texts:
        chunk_data = {
            "doc_id": f"doc{idx + 1}",  # Assign a document ID
            "chunk_id": str(uuid.uuid4()),  # Generate unique chunk ID
            "text": chunk
        }
        chunks.append(chunk_data)

# Create embeddings for each chunk
embeddings = model.encode([chunk['text'] for chunk in chunks])

# Convert embeddings to numpy array (FAISS requires NumPy arrays)
embeddings = np.array(embeddings)

# Initialize FAISS index (L2 distance metric)
index = faiss.IndexFlatL2(embeddings.shape[1])  # Ensure dimensionality matches the embeddings

# Add embeddings to FAISS index
index.add(embeddings)

# Save FAISS index to file
faiss.write_index(index, "data/index.faiss")

# Save chunks along with metadata (doc_id and chunk_id) in JSONL format for future reference
with open("data/chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk_data in chunks:
        f.write(json.dumps(chunk_data) + "\n")

print("Indexing completed and saved.")
