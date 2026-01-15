import pyarrow.parquet as pq
import faiss
import numpy as np
import pandas as pd
import json
from src.embeddings import EmbeddingExtractor # Import EmbeddingExtractor

class ADRFReader:
    def __init__(self, file_path: str, model_name: str = "openai/clip-vit-base-patch16"): # Added model_name
        self.file_path = file_path
        self.table = pq.read_table(file_path)
        self.df = self.table.to_pandas()
        
        # Initialize EmbeddingExtractor if needed for dynamic dimension
        self.model_name = model_name
        try:
            temp_extractor = EmbeddingExtractor(model_name=self.model_name)
            embed_dim = temp_extractor.model.config.projection_dim
        except Exception as e:
            print(f"Error initializing EmbeddingExtractor for {self.model_name}: {e}")
            embed_dim = 512 # Fallback dimension for CLIP base model

        # Load embeddings into FAISS index for searching
        if not self.df.empty:
            self.embeddings = np.stack(self.df['embedding'].values).astype('float32')
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            # Ensure index dimension matches actual embeddings.shape[1]
            if self.embeddings.shape[1] != embed_dim:
                print(f"Warning: Image embeddings dimension ({self.embeddings.shape[1]}) does not match model dimension ({embed_dim}).")
                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        else:
            self.embeddings = np.array([])
            self.index = None
        
        # Parse metadata back to dict if it was stored as json string
        if not self.df.empty and 'metadata' in self.df.columns and isinstance(self.df['metadata'].iloc[0], str):
            self.df['metadata'] = self.df['metadata'].apply(json.loads)

    def get_stats(self):
        """Returns basic stats about the loaded ADRF file."""
        return {
            "total_records": len(self.df),
            "columns": self.df.columns.tolist(),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings.size > 0 else 0
        }

    def search_similarity(self, query_embedding: np.ndarray, k=5):
        """Searches for the k nearest neighbors to the query embedding."""
        if self.index is None:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.df):
                record = self.df.iloc[idx].to_dict()
                record['similarity'] = float(dist)
                results.append(record)
        return results

    def get_record(self, index: int):
        if 0 <= index < len(self.df):
            return self.df.iloc[index].to_dict()
        return None

    def get_low_contribution_records(self, threshold: float = 0.2):
        """Returns records with contribution_score below the threshold."""
        if 'contribution_score' not in self.df.columns:
            return []
        
        filtered_df = self.df[self.df['contribution_score'] < threshold]
        return filtered_df.to_dict('records')