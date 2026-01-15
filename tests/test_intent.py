import sys
import os
import shutil
import json
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from src.reader import ADRFReader

def create_test_images(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create 10 "trees" (green squares)
    for i in range(10):
        img = Image.new('RGB', (224, 224), color=(0, 100+i*5, 0)) # Green variations
        img.save(os.path.join(output_dir, f"tree_{i}.png"))

def test_intent_curation():
    base_dir = "tests/test_data_intent"
    raw_dir = os.path.join(base_dir, "raw")
    output_file = os.path.join(base_dir, "output.adrf.parquet")
    stats_file = os.path.join(base_dir, "stats.json")
    preview_dir = os.path.join(base_dir, "preview")
    
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(raw_dir)

    create_test_images(raw_dir)
    
    print("\n--- Test 1: General Intent (Should mark redundant trees as unnecessary) ---")
    cmd = f"python adrf_convert.py --input_dir {raw_dir} --output_file {output_file} --stats_file {stats_file} --threshold 0.0001 --intent general"
    os.system(cmd)
    
    # Check results
    reader = ADRFReader(output_file)
    unnecessary_count = len(reader.df[reader.df['curation_action'] == 'unnecessary'])
    print(f"General Intent - Unnecessary Count: {unnecessary_count} / 10")
    
    # We expect high unnecessary count because they are redundant and not protected by alignment.
    # Note: Soft floor might save top 30% (3 images). So expect ~7 unnecessary.
    
    print("\n--- Test 2: Nature Intent (Should rescue trees) ---")
    # Clean output to force re-run logic completely or just update
    # If we update, logic re-runs on all data.
    # But let's delete parquet to be sure we start fresh scoring context if needed (though scoring is deterministic).
    if os.path.exists(output_file): os.remove(output_file)
    
    cmd = f"python adrf_convert.py --input_dir {raw_dir} --output_file {output_file} --stats_file {stats_file} --threshold 0.0001 --intent nature"
    os.system(cmd)
    
    reader = ADRFReader(output_file)
    # Check actions
    kept_count = len(reader.df[reader.df['curation_action'].isin(['keep', 'review', 'important'])])
    unnecessary_count_nature = len(reader.df[reader.df['curation_action'] == 'unnecessary'])
    
    print(f"Nature Intent - Kept/Review Count: {kept_count}")
    print(f"Nature Intent - Unnecessary Count: {unnecessary_count_nature}")
    
    # Sample reasons
    print("Sample Reasons:")
    print(reader.df[['curation_action', 'curation_reason']].head(3).to_string())
    
    if kept_count > 3: # More than just the soft floor
        print("SUCCESS: Intent 'nature' rescued redundant trees.")
    else:
        print("FAILURE: Intent did not rescue trees.")

if __name__ == "__main__":
    test_intent_curation()