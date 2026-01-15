import os
import shutil
import pandas as pd
from tqdm import tqdm
import json

def generate_preview(df: pd.DataFrame, output_dir: str):
    """
    Physically copies images into curation folders based on 'curation_action'.
    Non-destructive.
    """
    if df.empty:
        print("No data to preview.")
        return

    # Categories
    categories = ["keep", "important", "review", "unnecessary"]
    
    # Create folders
    for cat in categories:
        path = os.path.join(output_dir, cat)
        if not os.path.exists(path):
            os.makedirs(path)
            
    print(f"Generating preview in '{output_dir}'...")
    
    success_count = 0
    fail_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        action = row.get('curation_action', 'keep')
        if action not in categories: action = 'keep' # Fallback
        
        # Get source path
        # Metadata might be JSON string or dict
        meta = row.get('metadata')
        if isinstance(meta, str): meta = json.loads(meta)
        if not isinstance(meta, dict): continue
        
        src_path = meta.get('source_path')
        if not src_path or not os.path.exists(src_path):
            fail_count += 1
            continue
            
        file_name = os.path.basename(src_path)
        dest_path = os.path.join(output_dir, action, file_name)
        
        try:
            shutil.copy2(src_path, dest_path)
            success_count += 1
        except Exception as e:
            # print(f"Error copying {src_path}: {e}")
            fail_count += 1
            
    print(f"Preview generated: {success_count} copied, {fail_count} failed/missing.")
