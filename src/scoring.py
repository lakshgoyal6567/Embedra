import numpy as np
import faiss
from typing import Tuple, Dict

def calculate_contribution_scores(embeddings: np.ndarray, k=5) -> Tuple[np.ndarray, Dict]:
    """
    Calculates contribution scores based on uniqueness.
    Score is based on the average distance to the k nearest neighbors.
    Higher distance = Higher uniqueness = Higher score.
    Normalized to [0, 1] using 5th and 95th percentiles to be robust to outliers.
    
    Returns:
        scores: np.ndarray of shape (N,)
        metadata: Dict containing scaling parameters (p5, p95)
    """
    # Ensure embeddings is a numpy array of float32
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    if len(embeddings) == 0:
        return np.array([]), {}
        
    if len(embeddings) < 2:
        return np.ones(len(embeddings)), {"scaling_p5": 0.0, "scaling_p95": 0.0}

    # Build a temporary index for batch search
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype('float32'))
    
    # Search for k+1 neighbors (including self)
    k_search = min(len(embeddings), k + 1)
    distances, _ = index.search(embeddings.astype('float32'), k_search)
    
    # distances is (N, k_search). Column 0 is self (dist 0).
    if k_search > 1:
        neighbor_dists = distances[:, 1:]
        # Use mean distance to neighbors as the raw score
        raw_scores = np.mean(neighbor_dists, axis=1)
    else:
        raw_scores = np.zeros(len(embeddings))
        
    # Percentile-based Normalization
    p5 = np.percentile(raw_scores, 5)
    p95 = np.percentile(raw_scores, 95)
    
    # Avoid division by zero
    if p95 - p5 < 1e-9:
        # If range is too small, check if we have any variance at all
        if np.max(raw_scores) - np.min(raw_scores) > 1e-9:
            # Fallback to min-max if percentiles are collapsed but data isn't
            p5 = np.min(raw_scores)
            p95 = np.max(raw_scores)
        else:
            return np.ones(len(embeddings)) * 0.5, {"scaling_p5": float(p5), "scaling_p95": float(p95)}
        
    scores = (raw_scores - p5) / (p95 - p5)
    scores = np.clip(scores, 0.0, 1.0)
    
    metadata = {
        "scaling_p5": float(p5),
        "scaling_p95": float(p95),
        "raw_score_min": float(np.min(raw_scores)),
        "raw_score_max": float(np.max(raw_scores))
    }
    
    return scores, metadata