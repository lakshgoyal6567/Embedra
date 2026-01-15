import argparse
import subprocess
import sys
import os

def run_command(command):
    """Runs a shell command and streams output."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        process = subprocess.Popen(command, shell=True, cwd=script_dir, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Embedra CLI: The Sustainable AI Data Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- 1. PROCESS ALL (Local / Private) ---
    parser_process_all = subparsers.add_parser("process_all", help="Convert all raw data (images, docs, etc.) to ADRF (Local & Private)")
    parser_process_all.add_argument("--input_dir", type=str, default="data/raw_data", help="Path to directory containing all raw data (default: data/raw_data)")
    # Add common arguments here if needed, or pass them through to sub-converters

    # --- 2. UPLOAD (Cloud / Backup) ---
    parser_upload = subparsers.add_parser("upload", help="[OPT-IN] Upload vectors to Pinecone & images to S3")
    
    # --- 3. SEARCH (Local or Cloud) ---
    parser_search = subparsers.add_parser("search", help="Search your dataset")
    parser_search.add_argument("query", help="Text to search for")
    parser_search.add_argument("--mode", choices=["local", "cloud"], default="local", help="Search local file or Cloud Index")

    # --- 3. SEARCH DOCUMENTS ---
    parser_search_docs = subparsers.add_parser("search_docs", help="Search your ADRF Document Dataset")
    parser_search_docs.add_argument("query", type=str, help="Text query to search for")
    parser_search_docs.add_argument("--file", type=str, default="data/adrf/dataset_docs.adrf.parquet", help="ADRF Document Parquet file to search")
    parser_search_docs.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model name")
    parser_search_docs.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    parser_search_docs.add_argument("--min_score", type=float, default=0.20, help="Min confidence (0.0-1.0). Default: 0.20.")

    # --- 3. SEARCH MULTIMODAL ---
    parser_search_multimodal = subparsers.add_parser("search_multimodal", help="Search across multiple ADRF modalities (images, documents)")
    parser_search_multimodal.add_argument("query", type=str, help="Text query to search for")
    parser_search_multimodal.add_argument("--search_images", action="store_true", help="Include image ADRF in search")
    parser_search_multimodal.add_argument("--search_docs", action="store_true", help="Include document ADRF in search")
    parser_search_multimodal.add_argument("--image_file", type=str, default="data/adrf/dataset.adrf.parquet", help="Path to image ADRF file")
    parser_search_multimodal.add_argument("--doc_file", type=str, default="data/adrf/dataset_docs.adrf.parquet", help="Path to document ADRF file")
    parser_search_multimodal.add_argument("--model_image", type=str, default="openai/clip-vit-base-patch16", help="CLIP model for image embeddings")
    parser_search_multimodal.add_argument("--model_text", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model for text embeddings")
    parser_search_multimodal.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser_search_multimodal.add_argument("--min_score", type=float, default=0.20, help="Minimum confidence score for results")

    # Additional search arguments for multimodal search (e.g., --modality) can be added here later

    args = parser.parse_args()

    if args.command == "process_all":
        print(f"Embedra: Processing all raw data locally (Privacy Mode)...")
        # Call the new orchestrator script
        cmd = f"python src/processing/adrf_convert_all.py --input_dir \"{args.input_dir}\""
        run_command(cmd)
        print("\nProcessing complete. Data remains on your machine.")

    elif args.command == "upload":
        print("Embedra Cloud: Initiating Secure Upload...")
        print("   1. Syncing Search Index (Pinecone)...")
        run_command("python scripts/pinecone_sync.py --upload")
        
        print("\n   2. Syncing Image Vault (S3)...")
        run_command("python scripts/s3_upload.py")
        print("\nCloud Sync Complete. Your data is now backed up and searchable globally.")

    elif args.command == "search":
        if args.mode == "local":
            print(f"Searching Local ADRF (Confidence > 0.25)...")
            cmd = f"python scripts/demo_search.py --query \"{args.query}\""
            run_command(cmd)
        else:
            print(f"Searching Embedra Cloud...")
            cmd = f"python scripts/pinecone_sync.py --search_text \"{args.query}\""
            run_command(cmd)

    elif args.command == "search_docs":
        print(f"Searching Local Document ADRF...")
        cmd = f"python scripts/demo_search_docs.py \"{args.query}\" --file \"{args.file}\" --model \"{args.model}\" --top_k {args.top_k} --min_score {args.min_score}"
        run_command(cmd)
        print("\nDocument Search Complete.")

    elif args.command == "search_multimodal":
        print(f"Searching Multimodal ADRF...")
        # Construct the command for demo_search_multimodal.py
        cmd_parts = [
            "python", "scripts/demo_search_multimodal.py",
            f"\"{args.query}\""
        ]
        if args.search_images: cmd_parts.append("--search_images")
        if args.search_docs: cmd_parts.append("--search_docs")
        cmd_parts.append(f"--image_file \"{args.image_file}\"")
        cmd_parts.append(f"--doc_file \"{args.doc_file}\"")
        cmd_parts.append(f"--model_image \"{args.model_image}\"")
        cmd_parts.append(f"--model_text \"{args.model_text}\"")
        cmd_parts.append(f"--top_k {args.top_k}")
        cmd_parts.append(f"--min_score {args.min_score}")
        
        cmd = " ".join(cmd_parts)
        run_command(cmd)
        print("\nMultimodal Search Complete.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
