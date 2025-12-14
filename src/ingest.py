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
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks

# Read all .txt files from out_txt directory
texts = []
file_metadata = []

for file in os.listdir("data/out_txt"):
    if file.endswith(".txt"):
        with open(f"data/out_txt/{file}", "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(content)
            
            # Extract PDF name (remove .txt extension)
            pdf_name = file.replace(".txt", ".pdf")
            file_metadata.append({
                "pdf_name": pdf_name,
                "original_file": file
            })

# Chunk the text and create embeddings with metadata
chunks = []
for idx, text in enumerate(texts):
    chunked_texts = chunk_text(text)
    pdf_name = file_metadata[idx]["pdf_name"]
    
    for chunk_idx, chunk in enumerate(chunked_texts):
        # Estimate page number based on chunk position
        # Assuming ~2000 characters per page (rough estimate)
        estimated_page = (chunk_idx * (1000 - 150)) // 2000 + 1
        
        chunk_data = {
            "doc_id": f"doc{idx + 1}",
            "chunk_id": str(uuid.uuid4()),
            "text": chunk,
            "pdf_name": pdf_name,
            "page_number": estimated_page
        }
        chunks.append(chunk_data)

print(f"Created {len(chunks)} chunks from {len(texts)} documents")

# Create embeddings for each chunk
print("Encoding chunks...")
embeddings = model.encode([chunk['text'] for chunk in chunks], show_progress_bar=True)

# Convert embeddings to numpy array (FAISS requires NumPy arrays)
embeddings = np.array(embeddings)

# Initialize FAISS index (L2 distance metric)
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])

# Add embeddings to FAISS index
index.add(embeddings)

# Save FAISS index to file
faiss.write_index(index, "data/index.faiss")
print("FAISS index saved to data/index.faiss")

# Save chunks along with metadata (pdf_name, page_number, etc.) in JSONL format
with open("data/chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk_data in chunks:
        f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

print("Chunks with metadata saved to data/chunks.jsonl")
print("Indexing completed successfully!")