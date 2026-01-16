import os
import shutil
import glob

def clean_project():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    print(f"--- Cleaning project (Root: {project_root}) ---")

    # 1. Remove __pycache__ directories
    for dirpath, dirnames, filenames in os.walk(project_root):
        if "__pycache__" in dirnames:
            cache_path = os.path.join(dirpath, "__pycache__")
            print(f"Removing: {cache_path}")
            shutil.rmtree(cache_path)

    # 2. Remove .pytest_cache directories
    for dirpath, dirnames, filenames in os.walk(project_root):
        if ".pytest_cache" in dirnames:
            cache_path = os.path.join(dirpath, ".pytest_cache")
            print(f"Removing: {cache_path}")
            shutil.rmtree(cache_path)

    # 3. Remove generated ADRF files
    adrf_dir = os.path.join(project_root, "data", "adrf")
    if os.path.exists(adrf_dir):
        for f in glob.glob(os.path.join(adrf_dir, "*.adrf.parquet")):
            print(f"Removing: {f}")
            os.remove(f)

    # 4. Remove generated stats files
    summaries_dir = os.path.join(project_root, "data", "summaries")
    if os.path.exists(summaries_dir):
        stats_file = os.path.join(summaries_dir, "stats.json")
        if os.path.exists(stats_file):
            print(f"Removing: {stats_file}")
            os.remove(stats_file)

    # 5. Remove generated preview directories
    if os.path.exists(summaries_dir):
        preview_dir = os.path.join(summaries_dir, "preview") # Changed from 'curated_preview'
        if os.path.exists(preview_dir):
            print(f"Removing: {preview_dir}")
            shutil.rmtree(preview_dir)
        # Handle the default "curated_preview" if it still exists from old runs
        old_preview_dir = os.path.join(project_root, "curated_preview")
        if os.path.exists(old_preview_dir):
            print(f"Removing old preview dir: {old_preview_dir}")
            shutil.rmtree(old_preview_dir)

    print("--- Project cleanup complete ---")

if __name__ == "__main__":
    clean_project()