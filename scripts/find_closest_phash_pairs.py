import imagehash
from PIL import Image
import os
import glob
from tqdm import tqdm

def calculate_phash_for_all_images(image_dir):
    """Calculates pHash for all images in a directory."""
    image_hashes = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.avif', '*.tiff', '*.tif']
    
    # Collect all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    image_files = sorted(list(set(image_files)))
    
    for file_path in tqdm(image_files, desc="Calculating pHashes"):
        try:
            img = Image.open(file_path)
            # Ensure image is RGB for consistent hashing if library expects it
            if img.mode != 'RGB':
                img = img.convert('RGB')
            phash = imagehash.phash(img)
            image_hashes.append((os.path.basename(file_path), phash))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    return image_hashes

def find_furthest_phash_pairs(image_hashes, top_k=10):
    """Finds top_k pairs with the highest Hamming distance."""
    distances = []
    num_images = len(image_hashes)
    
    print(f"Comparing {num_images * (num_images - 1) // 2} pairs...")

    for i in tqdm(range(num_images), desc="Comparing pHashes"):
        for j in range(i + 1, num_images):
            file1, hash1 = image_hashes[i]
            file2, hash2 = image_hashes[j]
            
            distance = hash1 - hash2 # Hamming distance
            distances.append((distance, file1, file2))
            
    distances.sort(key=lambda x: x[0], reverse=True) # Sort by distance in descending order
    return distances[:top_k]

if __name__ == "__main__":
    raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_data"))
    
    print(f"Processing images in: {raw_data_dir}")
    
    all_image_hashes = calculate_phash_for_all_images(raw_data_dir)
    
    if not all_image_hashes:
        print("No images found or processed.")
    else:
        furthest_pairs = find_furthest_phash_pairs(all_image_hashes, top_k=10)
        
        print("\n--- Top 10 Image Pairs with Highest Hamming Distance ---")
        for distance, file1, file2 in furthest_pairs:
            print(f"Distance: {distance}, Files: {file1}, {file2}")
