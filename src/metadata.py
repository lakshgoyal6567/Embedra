import imagehash
from PIL import Image, ImageFilter, ImageStat
import os
import hashlib
from collections import Counter
import math
import numpy as np

def calculate_phash(image: Image.Image) -> str:
    """Computes the perceptual hash of an image."""
    return str(imagehash.phash(image))

def calculate_complexity(image: Image.Image) -> float:
    """
    Calculates visual complexity using edge density.
    Returns a score between 0.0 (empty) and 1.0 (highly complex).
    """
    try:
        # Convert to grayscale
        gray = image.convert('L')
        
        # Find edges
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy to count
        edge_data = np.array(edges)
        
        # Threshold for edge detection (ignore noise)
        threshold = 30
        edge_pixels = np.count_nonzero(edge_data > threshold)
        total_pixels = edge_data.size
        
        density = edge_pixels / total_pixels
        
        # Normalize: Density is rarely 1.0. A complex image might have 0.2 density.
        # Let's clip and scale so that >0.2 is "high" complexity (1.0).
        normalized = min(density * 5.0, 1.0)
        
        return float(normalized)
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return 0.0

def calculate_foreground_ratio(image: Image.Image) -> float:
    """
    Estimates the ratio of the image occupied by the foreground object
    by finding the bounding box of the edges.
    Returns float between 0.0 and 1.0.
    """
    try:
        # 1. Edge Detection
        gray = image.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_data = np.array(edges)
        
        threshold = 30
        edge_mask = edge_data > threshold
        
        if np.count_nonzero(edge_mask) == 0:
            return 0.0
            
        # 2. Find bounding box of edges
        y_indices, x_indices = np.where(edge_mask)
        if len(x_indices) == 0: return 0.0
        
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        bbox_area = bbox_width * bbox_height
        total_area = image.width * image.height
        
        ratio = bbox_area / total_area
        
        return float(ratio)
        
    except Exception as e:
        print(f"Error calculating foreground ratio: {e}")
        return 0.0

def get_dominant_color(image: Image.Image, size=(50, 50)) -> str:
    """
    Extracts the dominant color from an image.
    Returns hex code (e.g., '#FF0000').
    """
    try:
        image = image.copy()
        image.thumbnail(size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        colors = image.getcolors(maxcolors=size[0]*size[1])
        if not colors:
            return "#000000"
            
        most_frequent = max(colors, key=lambda item: item[0])[1]
        
        return '#{:02x}{:02x}{:02x}'.format(most_frequent[0], most_frequent[1], most_frequent[2])
    except Exception as e:
        print(f"Error extracting color: {e}")
        return "#000000"
        
def get_color_family(hex_color: str) -> str:
    """Maps a hex color to a broad family name."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # Simple heuristics
        if r < 40 and g < 40 and b < 40: return "black"
        if r > 215 and g > 215 and b > 215: return "white"
        if abs(r-g) < 20 and abs(r-b) < 20: return "gray"
        
        if r > g and r > b: return "red" if (g < 100 and b < 100) else "warm"
        if g > r and g > b: return "green"
        if b > r and b > g: return "blue"
        
        return "mixed"
    except:
        return "unknown"

def get_file_metadata(file_path: str) -> dict:
    """Extracts basic metadata from a file."""
    try:
        stat_info = os.stat(file_path)
        with Image.open(file_path) as img:
            width, height = img.size
            img_format = img.format
            mode = img.mode

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return {
            "file_name": os.path.basename(file_path),
            "file_size": stat_info.st_size,
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "checksum": sha256_hash.hexdigest(),
            "source_path": os.path.abspath(file_path)
        }
    except Exception as e:
        print(f"Error extracting metadata for {file_path}: {e}")
        return {}