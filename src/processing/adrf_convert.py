import argparse
import os
import glob
import hashlib
from tqdm import tqdm
from PIL import Image
import pillow_avif # Register AVIF support
import json
import numpy as np
import pandas as pd
import sys

# Add the project root to the Python path to enable imports from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Import modules
from src.embeddings import EmbeddingExtractor
from src.metadata import get_file_metadata, calculate_phash, get_dominant_color, get_color_family, calculate_complexity, calculate_foreground_ratio
from src.dedupe import Deduplicator
from src.stats import compute_dataset_stats, perform_clustering
from src.storage import save_to_parquet
from src.reader import ADRFReader
from src.scoring import calculate_contribution_scores
from src.curation import apply_coverage_driven_curation
from src.preview import generate_preview

# Candidate Labels for Zero-Shot Classification
LABELS_CONCEPTS = ["person", "animal", "vehicle", "food", "furniture", "building", "landscape", "electronics", "plant", "art", "other"]
LABELS_DOMAINS = ["nature", "urban", "indoor", "sketch", "text", "face", "medical", "satellite", "painting", "screenshot"]
LABELS_SCENES = ["indoor", "outdoor", "studio", "street", "forest", "beach", "office", "home", "highway", "sky"]

def get_complexity_level(complexity_score: float) -> str:
    """Maps complexity score (0-1) to low/medium/high."""
    if complexity_score < 0.15:
        return "low"
    elif complexity_score < 0.5:
        return "medium"
    else:
        return "high"

def detect_primitive_mode(semantic_results: dict, visual_complexity: float, score_threshold=0.5) -> bool:
    """
    Detects if an image is likely a 'primitive' (non-object) image.
    """
    concept_res = semantic_results.get('concept', {})
    if not concept_res: return False
    
    max_score = concept_res.get('score', 1.0)
    
    if max_score >= score_threshold:
        return False
    
    # Low confidence branch
    if visual_complexity >= 0.15:
        return False
    
    return True

def determine_abstraction_level(semantic_mode: str, semantic_results: dict) -> str:
    """
    Determines abstraction level: symbolic, object, or scene.
    """
    if semantic_mode == "primitive":
        return "symbolic"
    
    concept_score = semantic_results.get('concept', {}).get('score', 0.0)
    scene_score = semantic_results.get('scene', {}).get('score', 0.0)
    
    if concept_score >= scene_score:
        return "object"
    else:
        return "scene"

def main():
    parser = argparse.ArgumentParser(description="ADRF Converter: Raw Data to Model-Aware Format")
    parser.add_argument("--input_dir", type=str, default="data/raw_data", help="Path to directory containing images (default: data/raw_data)")
    parser.add_argument("--output_file", type=str, default="data/adrf/dataset.adrf.parquet", help="Path to output .adrf.parquet file (default: data/adrf/dataset.adrf.parquet)")
    parser.add_argument("--stats_file", type=str, default="data/summaries/stats.json", help="Path to output stats JSON file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Deduplication threshold (L2 distance)")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch16", help="HuggingFace model name")
    parser.add_argument("--preview_dir", type=str, default="data/summaries/preview", help="Directory to generate curation preview")
    parser.add_argument("--intent", type=str, default="general", help="Dataset intent (general, nature, autonomous, retail, synthetic)")
    parser.add_argument("--file_types", type=str, default=".jpg,.jpeg,.png,.bmp,.webp,.avif,.tiff,.tif", help="Comma-separated file extensions to process (e.g., '.jpg,.png')")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    # Check if there are any files to process
    valid_extensions = tuple(args.file_types.split(','))
    found_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                found_files.append(os.path.join(root, file))
    
    if not found_files:
        print(f"No images found in {args.input_dir} with extensions {args.file_types}. Skipping processing.")
        return

    try:
        extractor = EmbeddingExtractor(model_name=args.model)
        # Get dimension dynamically from model config
        embed_dim = extractor.model.config.projection_dim
        print(f"Model {args.model} loaded. Embedding Dimension: {embed_dim}")
        
        print("Pre-computing zero-shot classification embeddings...")
        text_embs_concepts = extractor.encode_text(LABELS_CONCEPTS)
        text_embs_domains = extractor.encode_text(LABELS_DOMAINS)
        text_embs_scenes = extractor.encode_text(LABELS_SCENES)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    deduplicator = Deduplicator(dimension=embed_dim, threshold=args.threshold)
    
    existing_df = pd.DataFrame()
    existing_checksums = set() # New: To store checksums of already processed files
    start_id = 0
    
    if os.path.exists(args.output_file):
        print(f"Found existing ADRF file: {args.output_file}. Updating...")
        try:
            reader = ADRFReader(args.output_file)
            existing_df = reader.df
            
            # New: Collect existing checksums from metadata
            if 'metadata' in existing_df.columns:
                for meta_str in existing_df['metadata']: # metadata might be stored as string
                    meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                    if 'checksum' in meta:
                        existing_checksums.add(meta['checksum'])

            if hasattr(reader, 'embeddings') and reader.embeddings.size > 0:
                print(f"Loading {len(reader.embeddings)} existing embeddings into deduplicator...")
                for emb in reader.embeddings:
                    deduplicator.add_embedding(emb)
            
            # Original existing_paths logic (removed as checksum is more robust)
            # if 'metadata' in existing_df.columns:
            #     for meta in existing_df['metadata']:
            #         if isinstance(meta, dict) and 'source_path' in meta:
            #             existing_paths.add(os.path.abspath(meta['source_path']))
            
            if not existing_df.empty and 'id' in existing_df.columns:
                start_id = existing_df['id'].max() + 1
                
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting fresh.")
            existing_df = pd.DataFrame()

    processed_data = []
    
    # Use file_types argument for filtering
    allowed_extensions = {ext.strip().lower() for ext in args.file_types.split(',')}
    
    image_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in allowed_extensions:
                image_files.append(os.path.join(root, file))
    
    image_files = sorted(list(set(image_files))) 
    
    print(f"Found {len(image_files)} images in input directory.")
    
    new_files_to_process = []
    for f in image_files:
        # New: Calculate SHA256 for the file
        sha256_hash_obj = hashlib.sha256()
        with open(f, "rb") as bf: # Use a different file object name to avoid conflict
            for byte_block in iter(lambda: bf.read(4096), b""):
                sha256_hash_obj.update(byte_block)
        file_checksum = sha256_hash_obj.hexdigest()
        
        if file_checksum in existing_checksums:
            # print(f"Skipping already processed file (checksum match): {os.path.basename(f)}")
            continue # Skip to next file
        
        # Original existing_paths logic (removed as checksum is more robust)
        # abs_path = os.path.abspath(f)
        # if abs_path in existing_paths:
        #     pass
        # else:
        new_files_to_process.append(f)
            
    print(f"Skipping {len(image_files) - len(new_files_to_process)} already processed files based on checksum or path.")
    print(f"Processing {len(new_files_to_process)} new or updated files...")
    
    current_id = start_id
    
    for file_path in tqdm(new_files_to_process):
        try:
            try:
                img = Image.open(file_path)
                img.load()
            except Exception as e:
                print(f"Skipping corrupt image {file_path}: {e}")
                continue

            metadata = get_file_metadata(file_path)
            phash = calculate_phash(img)
            embedding = extractor.extract_embedding(img)
            
            if embedding is None:
                continue
            
            is_dup, dist, match_idx = deduplicator.is_duplicate(embedding)
            
            if is_dup:
                continue
            
            deduplicator.add_embedding(embedding)
            
            # Semantic Classification
            # We use classify_subject which returns structured output
            res_concept = extractor.classify_subject(embedding, text_embs_concepts, LABELS_CONCEPTS)
            res_domain = extractor.classify_subject(embedding, text_embs_domains, LABELS_DOMAINS)
            res_scene = extractor.classify_subject(embedding, text_embs_scenes, LABELS_SCENES)
            
            # Derived Metrics
            complexity_score = calculate_complexity(img)
            complexity_level = get_complexity_level(complexity_score)
            foreground_ratio = calculate_foreground_ratio(img)
            
            is_primitive = detect_primitive_mode(
                {"concept": res_concept}, 
                visual_complexity=complexity_score,
                score_threshold=0.5
            )
            
            dominant_color = get_dominant_color(img)
            semantic_mode = "primitive" if is_primitive else "natural"
            
            abstraction_level = determine_abstraction_level(
                semantic_mode, 
                {"concept": res_concept, "scene": res_scene}
            )
            
            semantic_meta = {
                "semantic_mode": semantic_mode,
                "abstraction_level": abstraction_level,
                "visual_type": "solid_color" if is_primitive else None, 
                "dominant_color": dominant_color,
                "color_family": get_color_family(dominant_color) if is_primitive else None,
                "complexity": complexity_level,
                "complexity_score": float(complexity_score),
                "concept": res_concept if not is_primitive else None,
                "domain": res_domain if not is_primitive else None,
                "scene": res_scene if not is_primitive else None
            }
            
            semantic_meta = {k: v for k, v in semantic_meta.items() if v is not None}

            fact_meta = {
                "dominant_color": dominant_color,
                "object_count": 1 if abstraction_level == "object" else (0 if abstraction_level == "symbolic" else 2),
                "primary_object": res_concept["label"] if not is_primitive else None,
                "visual_complexity": complexity_level,
                "visual_complexity_score": float(complexity_score),
                "foreground_ratio": float(foreground_ratio)
            }
            
            # Calculate SHA256 for the file now, or re-use if already calculated
            # This logic is outside the file_path loop, so it's calculated once for every potential new file.
            # We need to make sure 'file_checksum' is available here.
            sha256_hash_obj = hashlib.sha256()
            with open(file_path, "rb") as f_hash:
                for byte_block in iter(lambda: f_hash.read(4096), b""):
                    sha256_hash_obj.update(byte_block)
            current_file_checksum = sha256_hash_obj.hexdigest()

            entry = {
                "id": int(current_id),
                "phash": phash,
                "metadata": metadata, # Contains source_path and original checksum
                "semantic_meta": semantic_meta,
                "fact_meta": fact_meta,
                "embedding": embedding,
                "contribution_score": 1.0,
                "checksum_original_file": current_file_checksum # Ensure consistent name
            }
            
            processed_data.append(entry)
            current_id += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Processing complete. New items: {len(processed_data)}.")

    if processed_data:
        new_df = pd.DataFrame(processed_data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = existing_df
    
    if not combined_df.empty:
        # ... (rest of the code)
        # Ensure 'metadata' column is not a string when accessing 'checksum'
        if 'metadata' in combined_df.columns:
            combined_df['metadata'] = combined_df['metadata'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        if 'checksum_original_file' not in combined_df.columns and 'metadata' in combined_df.columns:
            # Fallback for old records if checksum_original_file was not present as top-level
            combined_df['checksum_original_file'] = combined_df['metadata'].apply(lambda x: x.get('checksum'))
        
        # Drop metadata.checksum if it's redundant now
        if 'metadata' in combined_df.columns:
            combined_df['metadata'] = combined_df['metadata'].apply(lambda x: {k:v for k,v in x.items() if k != 'checksum'})
        
        # Final cleanup for combined_df
        # Re-convert metadata to string for Parquet storage if needed
        combined_df['metadata'] = combined_df['metadata'].apply(json.dumps)
        
        # --- Compute Stats & Save ---
        print("Computing dataset stats...")
        stats = compute_dataset_stats(combined_df['embedding'].tolist())
        
        print(f"Saving stats to {args.stats_file}")
        os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)
        with open(args.stats_file, 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"Saving ADRF dataset to {args.output_file}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        save_to_parquet(combined_df, args.output_file)
        
        print("Curation tasks...")
        scores, _ = calculate_contribution_scores(np.stack(combined_df['embedding'].to_numpy()))
        combined_df['contribution_score'] = scores
        
        curated_df = apply_coverage_driven_curation(combined_df, 0.5) # 50% target
        
        print("Generating preview (Curation Sort)...")
        os.makedirs(args.preview_dir, exist_ok=True)
        # Process all records for curation sorting
        generate_preview(curated_df, args.preview_dir)
        
        print("ADRF file and stats saved successfully.")

if __name__ == "__main__":
    main()
