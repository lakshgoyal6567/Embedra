import argparse
import os
import json
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.reader import ADRFReader
from src.embeddings import EmbeddingExtractor


def main():
    # Determine default path relative to script location to support running from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard structure: scripts/ -> root/ -> data/
    default_file = os.path.abspath(os.path.join(script_dir, "..", "data", "adrf", "dataset.adrf.parquet"))

    parser = argparse.ArgumentParser(description="Search your ADRF Dataset")
    parser.add_argument("--query", type=str, help="Text query to search for (e.g., 'a red car')")
    parser.add_argument("--image", type=str, help="Path to an image file to use as query")
    parser.add_argument("--file", type=str, default=default_file, help="ADRF Parquet file to search")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--min_score", type=float, default=0.25, help="Min confidence (0.0-1.0). Default: 0.25 (Good for Text). Use 0.70+ for Image-to-Image.")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch16", help="HuggingFace model name for embedding extraction.")
    
    args = parser.parse_args()
    
    if not args.query and not args.image:
        print("Error: You must provide either --query (text) or --image (file path).")
        return

    if not os.path.exists(args.file):
        print(f"Error: Dataset file '{args.file}' not found.")
        print("Did you run 'python adrf_convert.py' first?")
        return

    # 1. Load the Dataset
    print(f"Loading dataset from {args.file}...")
    try:
        reader = ADRFReader(args.file)
        stats = reader.get_stats()
        print(f"Loaded {stats['total_records']} records.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Initialize Model
    print("Loading AI model...")
    extractor = EmbeddingExtractor(model_name=args.model)
    
    # 3. Convert Query (Text or Image) to Embedding
    query_embedding = None
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            return
        print(f"Processing image query: {args.image}...")
        img = Image.open(args.image)
        query_embedding = extractor.extract_embedding(img)
    else:
        print(f"Processing text query: '{args.query}'...")
        # encode_text returns shape (1, 512)
        query_embedding = extractor.encode_text([args.query])
    
    if query_embedding is None or query_embedding.size == 0:
        print("Error creating embedding.")
        return

    # 4. Perform Search
    results = reader.search_similarity(query_embedding, k=args.top_k)

    
    print(f"\n--- Search Results (Min Confidence: {args.min_score:.2f}) ---")
    
    match_count = 0
    top_discarded = 0.0
    
    for i, res in enumerate(results):
        # FAISS with IndexFlatIP returns the inner product, which is the cosine similarity.
        # Higher is better.
        similarity = res.get('similarity', -1.0)
        
        if similarity < args.min_score:
            if similarity > top_discarded:
                top_discarded = similarity
            continue
            
        print(f"{match_count+1}. {format_result(res, similarity)}")
        match_count += 1
        
    if match_count == 0:
        print("No matches found within the confidence threshold.")
        print(f"Top discarded match had confidence: {top_discarded:.2%}")

def format_result(result, similarity):
    """Format a single search result for display."""
    meta = result.get('metadata', {})
    if isinstance(meta, str):
        meta = json.loads(meta)
    
    filename = os.path.basename(meta.get('source_path', 'Unknown'))
    
    sem_meta = result.get('semantic_meta', {})
    if isinstance(sem_meta, str): sem_meta = json.loads(sem_meta)
    
    concept = sem_meta.get('concept', {}).get('label', 'N/A')
    
    return f"File: {filename} | Concept: {concept} | Confidence: {similarity:.2%}"

if __name__ == "__main__":
    main()
