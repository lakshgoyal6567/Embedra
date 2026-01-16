import os
import sys
# Add parent directory to path for importing custom modules - MUST BE FIRST
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.embeddings import EmbeddingExtractor # Import EmbeddingExtractor

# Import existing ADRF readers
from src.reader import ADRFReader # For image ADRF
from src.reader_docs import ADRFDocReader # For document ADRF

def main():
    # Determine default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_image_file = os.path.abspath(os.path.join(script_dir, "..", "data", "adrf", "dataset.adrf.parquet"))
    default_doc_file = os.path.abspath(os.path.join(script_dir, "..", "data", "adrf", "dataset_docs.adrf.parquet"))

    parser = argparse.ArgumentParser(description="Multimodal ADRF Search")
    parser.add_argument("query", type=str, help="Text query to search for")
    parser.add_argument("--search_images", action="store_true", help="Search image ADRF")
    parser.add_argument("--search_docs", action="store_true", help="Search document ADRF")
    parser.add_argument("--image_file", type=str, default=default_image_file, help="Path to image ADRF file")
    parser.add_argument("--doc_file", type=str, default=default_doc_file, help="Path to document ADRF file")
    parser.add_argument("--model_image", type=str, default="openai/clip-vit-base-patch16", help="CLIP model for image embeddings")
    parser.add_argument("--model_text", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model for text embeddings")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--min_score", type=float, default=0.20, help="Minimum confidence score for results")
    
    args = parser.parse_args()

    if not args.search_images and not args.search_docs:
        print("Error: Please specify at least one modality to search (--search_images or --search_docs).")
        return

    # Initialize EmbeddingExtractor for image queries (to get 512-dim query embedding)
    print(f"Loading image embedding model for query: {args.model_image}...")
    try:
        query_image_extractor = EmbeddingExtractor(model_name=args.model_image)
    except Exception as e:
        print(f"Error loading image embedding model for query: {e}")
        return
    query_image_embedding = query_image_extractor.encode_text([args.query])
    if query_image_embedding.ndim == 1:
        query_image_embedding = query_image_embedding.reshape(1, -1)
    
    # Initialize SentenceTransformer for text queries (to get 384-dim query embedding for documents)
    print(f"Loading text embedding model for query: {args.model_text}...")
    try:
        query_text_model = SentenceTransformer(args.model_text)
    except Exception as e:
        print(f"Error loading text embedding model for query: {e}")
        return
    query_text_embedding = query_text_model.encode(args.query, convert_to_numpy=True)
    if query_text_embedding.ndim == 1:
        query_text_embedding = query_text_embedding.reshape(1, -1)

    all_results = []

    # Search Image ADRF
    if args.search_images:
        print(f"\n--- Searching Image ADRF ({args.image_file}) ---")
        if not os.path.exists(args.image_file):
            print(f"Image ADRF file not found: {args.image_file}. Skipping image search.")
        else:
            print(f"Loading image ADRF with model: {args.model_image}...")
            image_reader = ADRFReader(os.path.abspath(args.image_file), model_name=args.model_image)
            image_results = image_reader.search_similarity(query_image_embedding, k=args.top_k * 2) # Use 512-dim query embedding
            for res in image_results:
                if res['similarity'] >= args.min_score:
                    res['modality'] = 'image'
                    all_results.append(res)
            print(f"Found {len(image_results)} raw image results, {len([r for r in all_results if r['modality'] == 'image'])} filtered.")


    # Search Document ADRF
    if args.search_docs:
        print(f"\n--- Searching Document ADRF ({args.doc_file}) ---")
        if not os.path.exists(args.doc_file):
            print(f"Document ADRF file not found: {args.doc_file}. Skipping document search.")
        else:
            print(f"Loading document ADRF with model: {args.model_text}...")
            doc_reader = ADRFDocReader(os.path.abspath(args.doc_file), model_name=args.model_text)
            doc_results = doc_reader.search_similarity(query_text_embedding, k=args.top_k * 2) # Use 384-dim query embedding
            for res in doc_results:
                if res['similarity'] >= args.min_score:
                    res['modality'] = 'document'
                    all_results.append(res)
            print(f"Found {len(doc_results)} raw document results, {len([r for r in all_results if r['modality'] == 'document'])} filtered.")

    # Combine and Rank Results
    if all_results:
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"\n--- Top {args.top_k} Multimodal Results (Min Confidence: {args.min_score:.2f}) ---")
        for i, res in enumerate(all_results[:args.top_k]):
            print(f"{i+1}. Modality: {res['modality'].upper()}")
            print(f"   Confidence: {res['similarity']:.2%}")
            if res['modality'] == 'image':
                print(f"   File: {res['metadata'].get('file_name', 'N/A')} | Concept: {res['semantic_meta'].get('concept', {}).get('label', 'N/A')}")
            elif res['modality'] == 'document':
                print(f"   File: {res['original_filename']} | Chunk ID: {res['chunk_id']} | Page: {res.get('page_number', 'N/A')} | Snippet: {res['text_chunk'][:100]}...")
            print("-" * 20)
    else:
        print("\nNo multimodal results found within the confidence threshold.")


if __name__ == "__main__":
    main()