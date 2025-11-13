import faiss
import pickle
import numpy as np
from langchain.embeddings import OpenAIEmbeddings  # or HuggingFaceEmbeddings
import os

# --- Configuration ---
faiss_path = "/home/anik/Documents/AgenticAi/Lanchain_Practise/vector_store/faiss_index/index.faiss"
pkl_path   = "/home/anik/Documents/AgenticAi/Lanchain_Practise/vector_store/faiss_index/index.pkl"

# --- Check existence ---
if not os.path.exists(faiss_path):
    raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Metadata file not found at {pkl_path}")

# --- Load FAISS index ---
index = faiss.read_index(faiss_path)
print(f"✅ Index loaded\n  Dimension: {index.d}\n  Total vectors: {index.ntotal}")

# --- Load metadata ---
with open(pkl_path, "rb") as f:
    metadata = pickle.load(f)
print(f"✅ Metadata entries loaded: {len(metadata)}")

# --- Initialize embedding model ---
# You can replace this with any compatible model, e.g.:
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedder = OpenAIEmbeddings(model="text-embedding-3-small")  # requires OPENAI_API_KEY set in env

# --- Query ---
query_text = "How do payments work?"
query_emb = embedder.embed_query(query_text)
query_emb = np.array([query_emb]).astype("float32")  # FAISS requires 2D float32

# --- Search ---
k = 3
D, I = index.search(query_emb, k)
print(f"\n🔍 Query: {query_text}")
print(f"Top {k} Results:\n")

for rank, idx in enumerate(I[0]):
    doc = metadata[idx]
    print(f"[{rank+1}] Score: {D[0][rank]:.4f}")
    print(f"Text:\n{doc}\n{'-'*80}")

# --- Optional: Inspect stored vector (for debugging) ---
# vector = index.reconstruct(0)
# print("Example vector:", np.round(vector[:10], 4))
