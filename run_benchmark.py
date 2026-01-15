import subprocess
import json
import os
import re
import sys
import numpy as np

# Define benchmark queries and the complete set of relevant documents for each.
BENCHMARK_QUERIES = {
    "text_queries": [
        {"query": "a photo of a bike", "expected_files": ["images (18).jpg"]},
        {"query": "a photo of a cycle", "expected_files": ["download (19).jpg", "images (19).jpg", "images (21).jpg"]},
        {"query": "a photo of an apple", "expected_files": ["download (20).jpg", "images (28).jpg", "images (34).jpg"]},
        {"query": "a photo of a cat", "expected_files": ["download (27).jpg", "images (39).jpg", "images (41).jpg", "Cat1.jpg", "Cat2.jpg"]},
        {"query": "a photo of a dog", "expected_files": ["dog1.avif", "dog2.avif", "dog3.jpg"]},
        {"query": "a photo of a house", "expected_files": ["images (59).jpg", "House1.jpg", "House2.jpg", "House3.jpg", "House4.jpg"]},
        {"query": "a photo of a tree", "expected_files": ["Tree1.jpg", "Tree2.jpg", "Tree3.jpg", "Tree4.jpg"]},
        {"query": "a photo of a car", "expected_files": ["images (8).jpg", "Car1.jpg", "Car2.jpg", "Car3.avif", "Car4.avif", "download.jpg"]},
    ]
}

def run_search_query(query_text=None, min_score=0.20):
    """Runs demo_search.py and captures its output."""
    python_executable = sys.executable
    demo_search_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "demo_search.py")
    adrf_file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adrf", "dataset.adrf.parquet"))

    cmd_string = f'"{python_executable}" "{demo_search_script}" --model "openai/clip-vit-base-patch16" --file "{adrf_file_path}" --min_score {min_score} --top_k 20'
    if query_text:
        cmd_string += f' --query "{query_text}"'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process = subprocess.run(cmd_string, shell=True, cwd=script_dir, capture_output=True, text=True)
    return process.stdout

def parse_search_results(output):
    """Parses the output of demo_search.py to extract filenames."""
    results = []
    pattern = re.compile(r"^\d+\.\s+File: ([\w\d\s\/\-\._]+\.(?:jpg|jpeg|png|gif|bmp|avif|webp|tiff|tif))")
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            filename = os.path.basename(match.group(1))
            results.append(filename)
    return results

def calculate_average_precision(retrieved_files, expected_files):
    """Calculates Average Precision (AP) for a single query."""
    if not expected_files:
        return 1.0 if not retrieved_files else 0.0

    hits = 0
    precision_at_k = []
    for k, retrieved_file in enumerate(retrieved_files):
        if retrieved_file in expected_files:
            hits += 1
            precision_at_k.append(hits / (k + 1))

    if not precision_at_k:
        return 0.0

    return np.mean(precision_at_k)

def run_benchmark(thresholds):
    """Runs the benchmark for a list of thresholds and calculates mAP for each."""
    
    for threshold in thresholds:
        ap_scores = []
        print(f"--- Running Benchmark (mAP) with min_score = {threshold:.2f} ---")

        for query_data in BENCHMARK_QUERIES["text_queries"]:
            query = query_data["query"]
            expected_files = set(query_data["expected_files"])
            
            output = run_search_query(query_text=query, min_score=threshold)
            retrieved_files = parse_search_results(output)
            
            ap = calculate_average_precision(retrieved_files, expected_files)
            ap_scores.append(ap)

        mAP = np.mean(ap_scores) if ap_scores else 0.0
        print(f"\n--- Benchmark Complete for min_score = {threshold:.2f} ---")
        print(f"Mean Average Precision (mAP): {mAP:.2f}")

if __name__ == "__main__":
    thresholds_to_test = [0.15, 0.18, 0.20, 0.22]
    run_benchmark(thresholds_to_test)
