from PIL import Image, ImageFilter
import numpy as np

class StructureExtractor:
    def __init__(self):
        # Placeholder for DINOv2 or other structure models.
        # Currently uses heuristic edge-density based cropping.
        pass

    def extract_foreground(self, image: Image.Image) -> tuple[Image.Image, float]:
        """
        Extracts the foreground crop and estimates foreground ratio.
        
        Returns:
            (crop_image, foreground_ratio)
        """
        w, h = image.size
        
        # 1. Calculate Foreground Ratio (Edge Density Spread)
        # We reuse the logic previously in metadata.py but robustify it here
        gray = image.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_data = np.array(edges)
        threshold = 30
        edge_mask = edge_data > threshold
        
        total_edges = np.count_nonzero(edge_mask)
        if total_edges == 0:
            # Empty/Flat image -> Treat as full frame, low ratio
            return image, 0.0
            
        y_indices, x_indices = np.where(edge_mask)
        
        # Bounding Box of significant edges
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding (10%)
        pad_h = int(h * 0.1)
        pad_w = int(w * 0.1)
        
        y_min = max(0, y_min - pad_h)
        y_max = min(h, y_max + pad_h)
        x_min = max(0, x_min - pad_w)
        x_max = min(w, x_max + pad_w)
        
        # Crop Area
        crop_area = (x_max - x_min) * (y_max - y_min)
        total_area = w * h
        foreground_ratio = crop_area / total_area
        
        # Heuristic: If foreground is tiny (< 5%), falling back to center crop might be safer for CLIP
        # Or if it's huge (> 90%), just use full image
        
        if foreground_ratio < 0.05:
            # Likely noise or artifact. Use Center 50%
            left = w * 0.25
            top = h * 0.25
            right = w * 0.75
            bottom = h * 0.75
            crop = image.crop((left, top, right, bottom))
            return crop, 0.1
            
        if foreground_ratio > 0.9:
            # Full image is subject
            return image, 1.0
            
        # Perform Crop
        crop = image.crop((x_min, y_min, x_max, y_max))
        return crop, float(foreground_ratio)
