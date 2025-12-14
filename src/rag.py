import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2") 

# Load FAISS index and chunks
def load_index():
    """Load FAISS index and chunks - called once at startup"""
    try:
        index = faiss.read_index("data/index.faiss")
        with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f.readlines()]
        return index, chunks
    except FileNotFoundError:
        print("Warning: FAISS index or chunks file not found. Run ingest.py first.")
        return None, []

# Load once at module import
index, chunks = load_index()

def retrieve(query, k=3):  
    """Retrieve top K relevant chunks for the query (legacy function)"""
    if index is None or not chunks:
        return "No documents indexed. Please run ingest.py first."
    
    # Encode query
    query_embedding = model.encode([query])[0]
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Search FAISS index
    _, ids = index.search(query_embedding, k)
    
    # Check if results found
    if ids[0][0] == -1 or len(ids[0]) == 0:
        return "No relevant information found in the documents."
    
    # Combine top K chunks
    context = "\n\n".join([chunks[i]["text"] for i in ids[0] if i < len(chunks)])
    return context

def retrieve_with_metadata(query, k=3):
    """
    Retrieve top K relevant chunks with full metadata.
    
    Returns:
        list[dict]: List of source dictionaries with keys:
            - pdf_name (str): Name of the PDF
            - page_number (int): Page number in the PDF
            - content (str): The relevant text content
    """
    if index is None or not chunks:
        return []
    
    # Encode query
    query_embedding = model.encode([query])[0]
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Search FAISS index
    _, ids = index.search(query_embedding, k)
    
    # Check if results found
    if ids[0][0] == -1 or len(ids[0]) == 0:
        return []
    
    # Build source list with metadata
    sources = []
    for i in ids[0]:
        if i < len(chunks):
            chunk = chunks[i]
            sources.append({
                "pdf_name": chunk.get("pdf_name", "unknown.pdf"),
                "page_number": chunk.get("page_number", 1),
                "content": chunk.get("text", "")
            })
    
    return sources