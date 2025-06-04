# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import os
# import pickle

# def embed_and_store_chunks(chunks, model_name, index_path):
#     model = SentenceTransformer(model_name)
#     texts = [chunk["text"] for chunk in chunks]
#     doc_ids = [chunk["doc_id"] for chunk in chunks]

#     embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     # Save index and ID mapping
#     faiss.write_index(index, f"{index_path}.index")
#     with open(f"{index_path}_ids.pkl", "wb") as f:
#         pickle.dump(doc_ids, f)


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# 🔹 Model: all-MiniLM-L6-v2
# 🔹 Embedding size: 384 dimensions
# 🔹 Downloaded from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension of vectors produced by the model

def embed_and_store_chunks(chunks, index_path):
    """
    Embeds given chunks and stores FAISS index and ID mapping.
    """
    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Prepare data
    texts = [chunk["text"] for chunk in chunks]
    doc_ids = [chunk["doc_id"] for chunk in chunks]

    # Embed the texts
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, f"{index_path}.index")
    
    # Save mapping of index → doc_id
    with open(f"{index_path}_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"[✓] Stored {len(chunks)} embeddings to {index_path}.index")
