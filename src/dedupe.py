import faiss
import numpy as np

class Deduplicator:
    def __init__(self, dimension=512, threshold=0.1):
        self.dimension = dimension
        self.threshold = threshold
        # Using IndexFlatL2 for exact search. For very large datasets, consider IVFFlat.
        self.index = faiss.IndexFlatL2(dimension)
        self.stored_embeddings = [] # Keep track for ID mapping if needed, though FAISS handles IDs implicitly by insertion order

    def is_duplicate(self, embedding: np.ndarray):
        """
        Checks if the embedding is a duplicate of any existing embedding.
        Returns (is_duplicate, distance, index_of_match).
        """
        if self.index.ntotal == 0:
            return False, None, -1
        
        # FAISS expects float32 array of shape (n, d)
        query = embedding.reshape(1, -1).astype('float32')
        
        # Search for the nearest neighbor
        distances, indices = self.index.search(query, 1)
        
        distance = distances[0][0]
        index = indices[0][0]
        
        if distance < self.threshold:
            return True, distance, index
        return False, distance, index
    
    def add_embedding(self, embedding: np.ndarray):
        """Adds a new embedding to the index."""
        vector = embedding.reshape(1, -1).astype('float32')
        self.index.add(vector)