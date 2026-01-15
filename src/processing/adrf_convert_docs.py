import argparse
import os
import glob
import json
import re
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import hashlib

# Document Parsing Libraries
from pypdf import PdfReader # For PDF
from docx import Document # For DOCX
from sentence_transformers import SentenceTransformer # For embeddings

# Import custom modules (assuming they will be placed in src eventually)
# For now, we'll keep the EmbeddingExtractor in src for consistency with image
# But we will use SentenceTransformer directly here.

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    text = ""
    try:
        document = Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
    return text

def clean_text(text: str) -> str:
    """Cleans text by removing excessive whitespace and line breaks."""
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace with single space
    text = re.sub(r'(\n\s*){2,}', '\n\n', text) # Replace multiple blank lines with two newlines
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    """Splits text into chunks with a specified overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    parser = argparse.ArgumentParser(description="ADRF Converter: Document to Model-Aware Format")
    parser.add_argument("--input_dir", type=str, default="data/raw_docs", help="Path to directory containing documents (default: data/raw_docs)")
    parser.add_argument("--output_file", type=str, default="data/adrf/dataset_docs.adrf.parquet", help="Path to output .adrf.parquet file (default: data/adrf/dataset_docs.adrf.parquet)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model name")
    parser.add_argument("--chunk_size", type=int, default=250, help="Size of text chunks (in words)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between text chunks (in words)")
    parser.add_argument("--file_types", type=str, default=".txt,.pdf,.docx", help="Comma-separated file extensions to process (e.g., '.txt,.pdf')")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        os.makedirs(args.input_dir) # Create the directory if it doesn't exist
        print(f"Created empty directory: {args.input_dir}")
        return

    # Initialize Sentence Transformer model
    model = None # Initialize model to None
    print(f"Loading Sentence Transformer model: {args.model}...")
    try:
        model = SentenceTransformer(args.model)
        embed_dim = model.get_sentence_embedding_dimension()
        print(f"Model {args.model} loaded. Embedding Dimension: {embed_dim}")
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        # No return here, let the check after the block handle it.

    if model is None: # Explicitly check if model was loaded
        print("Failed to load Sentence Transformer model. Exiting.")
        return

    existing_df = pd.DataFrame()
    existing_checksums = set() # New: To store checksums of already processed files
    
    if os.path.exists(args.output_file):
        print(f"Found existing Document ADRF file: {args.output_file}. Updating...")
        try:
            reader = pd.read_parquet(args.output_file) # Directly read with pandas
            existing_df = reader
            if 'checksum_original_file' in existing_df.columns:
                existing_checksums.update(existing_df['checksum_original_file'].unique())
            print(f"Loaded {len(existing_df)} existing records.")
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting fresh.")
            existing_df = pd.DataFrame()

    processed_data = []
    
    # Use file_types argument for filtering
    allowed_extensions = {ext.strip().lower() for ext in args.file_types.split(',')}
    
    document_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in allowed_extensions:
                document_files.append(os.path.join(root, file))
    
    document_files = sorted(list(set(document_files)))

    print(f"Found {len(document_files)} documents in input directory.")

    new_files_to_process = []
    for f in document_files:
        sha256_hash_obj = hashlib.sha256()
        with open(f, "rb") as bf:
            for byte_block in iter(lambda: bf.read(4096), b""):
                sha256_hash_obj.update(byte_block)
        file_checksum = sha256_hash_obj.hexdigest()

        if file_checksum in existing_checksums:
            print(f"Skipping already processed document (checksum match): {os.path.basename(f)}")
        else:
            new_files_to_process.append(f)
            
    print(f"Processing {len(new_files_to_process)} new or updated documents.")

    for file_path in tqdm(new_files_to_process, desc="Processing documents"):
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1].lower()
        
        raw_text = ""
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
        elif file_extension == '.pdf':
            raw_text = extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            raw_text = extract_text_from_docx(file_path)
        else:
            print(f"Skipping unsupported file type: {file_name}")
            continue

        if not raw_text.strip():
            print(f"Skipping empty text from {file_name}")
            continue
        
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text, args.chunk_size, args.overlap)

        # Generate SHA256 for the original file
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue # Skip empty chunks

            embedding = model.encode(chunk, convert_to_numpy=True)
            
            # Metadata for this chunk
            entry = {
                "original_filename": file_name,
                "file_extension": file_extension,
                "chunk_id": i,
                "text_chunk": chunk,
                "embedding": embedding.tolist(), # Store as list in Parquet
                "checksum_original_file": sha256_hash.hexdigest(),
                "source_path_original_file": os.path.abspath(file_path)
            }
            processed_data.append(entry)
            
    if processed_data:
        df = pd.DataFrame(processed_data)
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        df.to_parquet(args.output_file, index=False)
        print(f"\nSuccessfully saved {len(df)} text chunks to {args.output_file}")
    else:
        print("No documents processed or no text chunks found.")

if __name__ == "__main__":
    main()