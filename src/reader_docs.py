import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

class ADRFDocReader:
    def __init__(self, file_path: str, model_name: str = "all-MiniLM-L6-v2"): # Added model_name
        self.file_path = file_path
        self.df = pd.read_parquet(file_path)
        self.model_name = model_name # Store model name

        # Initialize Sentence Transformer model for query embedding (if not already done)
        # This will be used when search_similarity is called by demo_search_multimodal.
        # But for search_similarity, the query is already an embedding.
        # So the model is only needed here to get embed_dim for FAISS.
        try:
            temp_model = SentenceTransformer(self.model_name)
            embed_dim = temp_model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Error loading Sentence Transformer model {self.model_name}: {e}")
            embed_dim = 384 # Fallback
        
        # Load embeddings into FAISS index for searching
        if not self.df.empty:
            self.embeddings = np.stack(self.df['embedding'].values).astype('float32')
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1]) # Use IP for cosine similarity
            # Ensure index dimension matches actual embeddings.shape[1]
            if self.embeddings.shape[1] != embed_dim:
                print(f"Warning: Embeddings dimension ({self.embeddings.shape[1]}) does not match model dimension ({embed_dim}).")
                # Adjust index if necessary, or error out
                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        else:
            self.embeddings = np.array([])
            self.index = None
        
    def search_similarity(self, query_embedding: np.ndarray, k=5):
        """Searches for the k nearest neighbors to the query embedding."""
        if self.index is None:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure query embedding is also normalized for cosine similarity with IP index
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.df):
                record = self.df.iloc[idx].to_dict()
                record['similarity'] = float(dist) # Store inner product as similarity
                results.append(record)
        return results
