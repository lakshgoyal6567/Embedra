import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Dict, Tuple

class EmbeddingExtractor:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extracts the embedding for a single image."""
        try:
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Move to CPU and convert to numpy
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of text labels into normalized embeddings."""
        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()
        except Exception as e:
            print(f"Error encoding text: {e}")
            return np.array([])

    def classify_subject(self, image_emb: np.ndarray, text_embs: np.ndarray, labels: List[str]) -> Dict:
        """
        Classifies the primary subject.
        Returns: {
            "label": str,
            "score": float,
            "alternatives": List[str],
            "certainty": str (high, moderate, low)
        }
        """
        if image_emb is None or len(text_embs) == 0:
            return {"label": "unknown", "score": 0.0, "alternatives": [], "certainty": "low"}
        
        # Cosine similarity
        similarities = np.dot(text_embs, image_emb)
        
        logit_scale = self.model.logit_scale.exp().item()
        logits = similarities * logit_scale
        
        probs = np.exp(logits - np.max(logits)) 
        probs = probs / probs.sum()
        
        # Get Top 3
        top_indices = np.argsort(probs)[-3:][::-1]
        
        best_idx = top_indices[0]
        score = float(probs[best_idx])
        
        alternatives = []
        if len(top_indices) > 1:
            for idx in top_indices[1:]:
                if probs[idx] > 0.1: # Only include decent alternatives
                    alternatives.append(labels[idx])
        
        # Determine Certainty
        margin = 0.0
        if len(top_indices) > 1:
            margin = score - float(probs[top_indices[1]])
        else:
            margin = score
            
        certainty = "low"
        if score > 0.6 and margin > 0.2:
            certainty = "high"
        elif score > 0.4:
            certainty = "moderate"
            
        return {
            "label": labels[best_idx],
            "score": score,
            "alternatives": alternatives,
            "certainty": certainty
        }
