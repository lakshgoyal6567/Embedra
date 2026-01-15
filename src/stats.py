import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans

def perform_clustering(embeddings: np.ndarray, min_k=3, max_k=50) -> Tuple[np.ndarray, int]:
    """
    Performs K-Means clustering.
    Determines k dynamically: roughly sqrt(N/2), clamped between min_k and max_k.
    Returns (labels, k_used).
    """
    n = len(embeddings)
    if n < min_k:
        # Too few items to cluster effectively, treat as 1 cluster (or N clusters?)
        # Let's treat as 1 cluster (label 0 for all)
        return np.zeros(n, dtype=int), 1
        
    # Dynamic K
    # Heuristic: We want clusters to have ~5-10 items on average for curation to meaningful.
    # But for small datasets (18 items), k=3 or 4 is fine.
    # Rule of thumb: sqrt(n)
    k = int(np.sqrt(n))
    k = max(min_k, min(k, max_k, n)) # Clamp
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    return labels, k

def compute_dataset_stats(embeddings: List[np.ndarray], labels: List[str] = None, contribution_scores: Optional[List[float]] = None) -> Dict:
    """Computes statistical summaries of the dataset."""
    if not embeddings:
        return {"error": "No embeddings provided"}
    
    matrix = np.array(embeddings)
    
    # Mean vector
    mean_vector = np.mean(matrix, axis=0)
    
    # Distance distribution (pairwise is expensive, so maybe dist to mean)
    dists_to_mean = np.linalg.norm(matrix - mean_vector, axis=1)
    
    stats = {
        "count": len(embeddings),
        "embedding_dim": matrix.shape[1],
        "mean_vector_norm": float(np.linalg.norm(mean_vector)),
        "dist_to_mean_avg": float(np.mean(dists_to_mean)),
        "dist_to_mean_std": float(np.std(dists_to_mean)),
        "dists_to_mean_min": float(np.min(dists_to_mean)),
        "dists_to_mean_max": float(np.max(dists_to_mean)),
    }
    
    # Contribution Scores Stats
    if contribution_scores is not None and len(contribution_scores) > 0:
        scores = np.array(contribution_scores)
        stats["contribution_score_mean"] = float(np.mean(scores))
        stats["contribution_score_min"] = float(np.min(scores))
        stats["contribution_score_max"] = float(np.max(scores))
        stats["contribution_score_std"] = float(np.std(scores))
    
    # Optional: Cluster centroids (e.g., k=5 for summary)
    if len(embeddings) > 5:
        kmeans = KMeans(n_clusters=min(5, len(embeddings)), random_state=42, n_init='auto')
        kmeans.fit(matrix)
        stats["cluster_centers_sample"] = kmeans.cluster_centers_.tolist()

    if labels:
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        stats["label_distribution"] = label_counts
        
    return stats