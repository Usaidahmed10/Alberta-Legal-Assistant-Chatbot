import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {
    "excel_path": os.path.join(BASE_DIR, "data", "alberta_laws.xlsx"),
    "pdf_dir": os.path.join(BASE_DIR, "data", "raw_pdfs"),
    "text_dir": os.path.join(BASE_DIR, "data", "processed_texts"),
    "vector_store_path": os.path.join(BASE_DIR, "embeddings", "faiss_index"),
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size_words": 200,
    "top_k_docs": 3,
    "top_k_chunks": 5
}
