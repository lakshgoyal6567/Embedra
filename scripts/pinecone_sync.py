import argparse
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.reader import ADRFReader
from src.embeddings import EmbeddingExtractor
from PIL import Image
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("Please install pinecone-client: pip install pinecone-client")
    exit(1)

# --- CONFIGURATION ---
API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY or API_KEY == "YOUR_PINECONE_API_KEY_HERE":
    print("Error: PINECONE_API_KEY not found or not set in .env file.")
    print("Please create a .env file and set your API key.")
    exit(1)

INDEX_NAME = "project_name-index-large"
DIMENSION = 768
METRIC = "cosine" # CLIP works best with Cosine
CLOUD = "aws" 
REGION = "us-east-1" # Free tier is often us-east-1

def setup_index(pc):
    """Creates the index if it doesn't exist."""
    print(f"Checking index '{INDEX_NAME}'...")
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new index '{INDEX_NAME}'...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud=CLOUD,
                    region=REGION
                )
            )
            print("Index created. Waiting for initialization...")
            time.sleep(10) # Wait for index to be ready
        except Exception as e:
            print(f"Error creating index: {e}")
            return None
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
        
    return pc.Index(INDEX_NAME)

def upload_vectors(index, parquet_file):
    """Reads vectors from Parquet and uploads to Pinecone."""
    print(f"Loading data from {parquet_file}...")
    reader = ADRFReader(parquet_file)
    df = reader.df
    
    print(f"Found {len(df)} records. Preparing upload...")
    
    batch_size = 100
    vectors_to_upsert = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Metadata filtering logic
        # Pinecone metadata values must be str, int, float, or list of strings
        # We flatten our complex metadata
        meta = row.get('metadata', {})
        if isinstance(meta, str): import json; meta = json.loads(meta)
        
        sem_meta = row.get('semantic_meta', {})
        if isinstance(sem_meta, str): import json; sem_meta = json.loads(sem_meta)
        
        # Construct clean metadata for Pinecone
        clean_meta = {
            "filename": os.path.basename(meta.get('source_path', 'unknown')),
            "concept": sem_meta.get('concept', {}).get('label', 'unknown') if sem_meta else 'unknown',
            "score": float(row.get('contribution_score', 0.0))
        }
        
        # ID must be string
        vec_id = str(row['id'])
        
        # Vector must be list of floats
        vector = row['embedding']
        if hasattr(vector, 'tolist'): vector = vector.tolist()
        
        vectors_to_upsert.append({
            "id": vec_id,
            "values": vector,
            "metadata": clean_meta
        })
        
        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert = []
            
    # Final batch
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        
    print("Upload complete.")

def search_pinecone(index, query_text=None, image_path=None):
    """Searches the Pinecone index."""
    print("\n--- Testing Cloud Search ---")
    extractor = EmbeddingExtractor()
    query_vec = []
    
    if image_path:
        print(f"Processing image: {image_path}")
        img = Image.open(image_path)
        query_vec = extractor.extract_embedding(img).tolist()
    elif query_text:
        print(f"Processing text: '{query_text}'")
        query_vec = extractor.encode_text([query_text])[0].tolist()
        
    if not query_vec:
        print("No query vector generated.")
        return

    print("Querying Pinecone...")
    results = index.query(
        vector=query_vec,
        top_k=3,
        include_metadata=True
    )
    
    for match in results['matches']:
        print(f"ID: {match['id']} | Score: {match['score']:.4f} | File: {match['metadata']['filename']}")

def main():
    parser = argparse.ArgumentParser(description="project_name Cloud Sync (Pinecone)")
    parser.add_argument("--upload", action="store_true", help="Upload data from parquet to Pinecone")
    parser.add_argument("--search_text", type=str, help="Search cloud with text")
    parser.add_argument("--search_image", type=str, help="Search cloud with image")
    parser.add_argument("--file", type=str, default="output.adrf.parquet", help="Parquet file")
    
    args = parser.parse_args()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=API_KEY)
    index = setup_index(pc)
    
    if not index:
        return

    if args.upload:
        if os.path.exists(args.file):
            upload_vectors(index, args.file)
        else:
            print("Parquet file not found.")
            
    if args.search_text:
        search_pinecone(index, query_text=args.search_text)
        
    if args.search_image:
        search_pinecone(index, image_path=args.search_image)

if __name__ == "__main__":
    main()
