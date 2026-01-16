import argparse
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ADRFDocReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_parquet(file_path)
        
        # Load embeddings into FAISS index for searching
        if not self.df.empty:
            self.embeddings = np.stack(self.df['embedding'].values).astype('float32')
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1]) # Use IP for cosine similarity
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

def main():
    # Determine default path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.abspath(os.path.join(script_dir, "..", "data", "adrf", "dataset_docs.adrf.parquet"))

    parser = argparse.ArgumentParser(description="Search your ADRF Document Dataset")
    parser.add_argument("query", type=str, help="Text query to search for")
    parser.add_argument("--file", type=str, default=default_file, help="ADRF Document Parquet file to search")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model name")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--min_score", type=float, default=0.20, help="Min confidence (0.0-1.0). Default: 0.20.")
    
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: Document ADRF file '{args.file}' not found.")
        print("Did you run 'python project_name.py process_all' first?")
        return

    # 1. Load the Dataset
    print(f"Loading document dataset from {args.file}...")
    try:
        reader = ADRFDocReader(os.path.abspath(args.file))
        print(f"Loaded {len(reader.df)} document chunks.")
    except Exception as e:
        print(f"Failed to load document dataset: {e}")
        return

    # 2. Initialize Sentence Transformer model
    print(f"Loading Sentence Transformer model: {args.model}...")
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        return

    # 3. Convert Query to Embedding
    query_embedding = model.encode(args.query, convert_to_numpy=True)
    
    if query_embedding is None or query_embedding.size == 0:
        print("Error creating query embedding.")
        return

    # Ensure query embedding is 2D for FAISS
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # 4. Perform Search
    results = reader.search_similarity(query_embedding, k=args.top_k)
    
    print(f"\n--- Search Results (Min Confidence: {args.min_score:.2f}) ---")
    
    match_count = 0
    top_discarded = 0.0
    
    for i, res in enumerate(results):
        similarity = res.get('similarity', -1.0) # Using 'similarity' key now
        
        if similarity < args.min_score:
            if similarity > top_discarded:
                top_discarded = similarity
            continue
            
        print(f"{match_count+1}. File: {res['original_filename']} | Chunk ID: {res['chunk_id']} | Page: {res.get('page_number', 'N/A')} | Confidence: {similarity:.2%}")
        print(f"   Snippet: {res['text_chunk'][:200]}...") # Print a snippet
        match_count += 1
        
    if match_count == 0:
        print("No matches found within the confidence threshold.")
        print(f"Top discarded match had confidence: {top_discarded:.2%}")

if __name__ == "__main__":
    main()
