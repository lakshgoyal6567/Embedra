from typing import Dict, Any, List, Tuple
import pandas as pd
import json
import numpy as np

# Intent Profiles (Percent-based)
INTENT_PROFILES = {
    "general": {
        "description": "Balanced coverage: 50% Important, 80% Keep.",
        "min_coverage_pct": 0.50, 
        "keep_pct": 0.80,
        "priority_categories": [] 
    },
    "nature": {
        "description": "Focus on natural world.",
        "min_coverage_pct": 0.30, 
        "keep_pct": 0.50,
        "priority_categories": ["nature", "animal"]
    },
    "autonomous": {
        "description": "Focus on vehicles and structures.",
        "min_coverage_pct": 0.30,
        "keep_pct": 0.50,
        "priority_categories": ["vehicle", "person", "structure"]
    },
    "retail": {
        "description": "Focus on objects.",
        "min_coverage_pct": 0.30,
        "keep_pct": 0.50,
        "priority_categories": ["object", "art"]
    },
    "synthetic": {
        "description": "Focus on art.",
        "min_coverage_pct": 0.30,
        "keep_pct": 0.50,
        "priority_categories": ["art"]
    }
}

# Mapping only used for Intent Priority checks now, not for grouping
CANONICAL_MAPPING = {
    "person": "person",
    "animal": "animal",
    "vehicle": "vehicle",
    "food": "object",
    "furniture": "object",
    "electronics": "object",
    "building": "structure",
    "landscape": "nature",
    "plant": "nature",
    "art": "art",
    "other": "misc"
}

def get_canonical_category(record: Dict[str, Any]) -> str:
    """Derives canonical category from semantic metadata."""
    sem_meta = record.get('semantic_meta', {})
    if isinstance(sem_meta, str): sem_meta = json.loads(sem_meta)
    
    mode = sem_meta.get('semantic_mode', 'natural')
    if mode == 'primitive': return "texture"
    
    subject = sem_meta.get('primary_subject', {})
    if subject and isinstance(subject, dict) and 'label' in subject:
        label = subject.get('label', 'other')
    else:
        label = sem_meta.get('concept', {}).get('label', 'other')
        
    return CANONICAL_MAPPING.get(label, "misc")

def calculate_quality_score(row: pd.Series) -> float:
    """
    Calculates a quality score based on a nuanced combination of factors.
    """
    contrib = row.get('contribution_score', 0.0)
    
    sem_meta = row.get('semantic_meta', {})
    if isinstance(sem_meta, str): sem_meta = json.loads(sem_meta)
    
    subject = sem_meta.get('primary_subject', {})
    conf = subject.get('score', 0.5) # Default confidence if not found
    
    complexity = sem_meta.get('complexity_score', 0.5)
    foreground_ratio = sem_meta.get('foreground_ratio', 0.0)
    abstraction_level = sem_meta.get('abstraction_level', 'unknown')

    # 1. Base Score: Weighted average of contribution and confidence
    base_score = (0.5 * contrib) + (0.5 * conf) # Increased weight for core relevance

    # 2. Abstraction Level Bonus/Penalty
    if abstraction_level == 'object':
        base_score += 0.15 # Strong bonus for clear objects
    elif abstraction_level == 'scene':
        base_score += 0.05 # Smaller bonus for scenes (often less focused)
    elif abstraction_level == 'symbolic':
        base_score -= 0.10 # Penalty for symbolic/primitive images (less useful for general search)

    # 3. Foreground Ratio Bonus (reward images with more prominent foregrounds)
    base_score += (0.10 * foreground_ratio) 

    # 4. Complexity Sweet Spot Bonus
    # Images with moderate complexity (e.g., 0.4-0.7) are often more "interesting" and useful
    if 0.4 <= complexity <= 0.7:
        base_score += 0.05
    elif complexity < 0.2 or complexity > 0.8: # Very simple or very complex
        base_score -= 0.05

    # Ensure score stays within reasonable bounds [0, 1]
    return np.clip(base_score, 0.0, 1.0)

def is_critically_ambiguous(row: pd.Series) -> Tuple[bool, str]:
    sem_meta = row.get('semantic_meta', {})
    if isinstance(sem_meta, str): sem_meta = json.loads(sem_meta)
    
    subject = sem_meta.get('primary_subject', {})
    conf = subject.get('score', 1.0)
    certainty = subject.get('certainty', 'moderate')
    
    if conf < 0.35:
        return True, f"Critically low confidence ({conf:.2f})"
    if certainty == 'low' and conf < 0.5:
        return True, "Low certainty & confidence"
    return False, ""

def apply_coverage_driven_curation(df: pd.DataFrame, intent_name: str = "general") -> pd.DataFrame:
    """
    Applies Percentage-Based Curation Logic using CLUSTERS instead of categories.
    """
    if df.empty: return df
    
    intent = INTENT_PROFILES.get(intent_name, INTENT_PROFILES["general"])
    priority_cats = set(intent.get("priority_categories", []))
    
    df['dataset_intent'] = intent_name
    
    # Pre-calculate Canonical Categories (for Intent Priority Check only)
    canonical_cats = []
    quality_scores = []
    
    for idx, row in df.iterrows():
        canonical_cats.append(get_canonical_category(row.to_dict()))
        quality_scores.append(calculate_quality_score(row))
        
    df['canonical_category'] = canonical_cats
    df['quality_score'] = quality_scores
    
    curation_actions = [""] * len(df)
    curation_reasons = [""] * len(df)
    coverage_statuses = [""] * len(df)
    
    # Group by CLUSTER_ID AND CANONICAL_CATEGORY
    # This prevents semantic contamination (e.g. a 'misc' image in a 'nature' cluster stealing the anchor spot)
    
    # Create a grouping key
    if 'cluster_id' not in df.columns:
        print("Warning: cluster_id not found. Falling back to single group.")
        df['cluster_id'] = 0
        
    # Get unique combinations of (cluster, category)
    groups = df.groupby(['cluster_id', 'canonical_category']).groups
    
    for (cid, cat), indices in groups.items():
        # Indices is already the list of index labels
        cluster_indices = indices
        
        # Sort by Quality
        sorted_indices = df.loc[cluster_indices].sort_values('quality_score', ascending=False).index
        count = len(sorted(indices))
        
        # Determine Thresholds
        # Priority check is now direct on the category of this sub-group
        is_priority = cat in priority_cats
        
        pct_important = intent["min_coverage_pct"]
        pct_keep = intent["keep_pct"]
        
        if is_priority:
            pct_important = min(1.0, pct_important * 1.5)
            pct_keep = min(1.0, pct_keep * 1.5)
            
        n_important = max(1, int(np.ceil(count * pct_important)))
        n_keep = int(np.ceil(count * pct_keep))
        
        for rank, idx in enumerate(sorted_indices):
            row = df.loc[idx]
            
            is_critical, critical_reason = is_critically_ambiguous(row)
            # Redundancy Check
            is_redundant = (row['contribution_score'] < 0.15) and (not is_priority)
            
            # Debug
            # if cat == "nature":
            #    print(f"DEBUG: Cat={cat}, Rank={rank}, Score={row['contribution_score']:.3f}, n_imp={n_important}, Redundant={is_redundant}, Critical={is_critical}")
            
            if is_critical:
                action = "review"
                status = "uncertain"
                reason_str = critical_reason
            elif is_redundant and rank >= n_important:
                # Redundant AND outside the Important anchor slot
                # BUT wait, the user asked to "Keep All Valid" (remove unecessary unless garbage).
                # My previous change was:
                # elif is_redundant: (with removed rank check) -> This was too harsh.
                # Then I did: "Keep Everything else that is valid".
                
                # REVERTED LOGIC FROM PREVIOUS TURN:
                # "I will update src/curation.py to implement the 'Keep All Valid' policy."
                # "Unnecessary: Only assigned if contribution_score < 0.15"
                # But here, 'is_redundant' IS score < 0.15.
                # So if score < 0.15, we mark UNNECESSARY.
                # Unless it is Rank < n_important (The Anchor).
                
                # So: if is_redundant and rank >= n_important -> UNNECESSARY.
                # This is correct.
                # Rank 0 (Anchor) is kept even if score < 0.15.
                # Rank 1+ (Duplicates) are discarded if score < 0.15.
                
                action = "unnecessary"
                status = "excess"
                reason_str = "Redundant (Score < 0.15)"
            else:
                if rank < n_important:
                    action = "important"
                    status = "satisfied"
                    reason_str = f"Top 20% (Rank {rank+1}/{count})"
                    if is_priority: reason_str += " [Priority]"
                else:
                    action = "keep"
                    status = "satisfied"
                    if rank < n_important + n_keep:
                        reason_str = f"Next 40% (Rank {rank+1}/{count})"
                    else:
                        reason_str = f"Valid excess (Rank {rank+1}/{count})"
                    if is_priority: reason_str += " [Priority]"
            
            curation_actions[df.index.get_loc(idx)] = action
            curation_reasons[df.index.get_loc(idx)] = reason_str
            coverage_statuses[df.index.get_loc(idx)] = status

    df['curation_action'] = curation_actions
    df['curation_reason'] = curation_reasons
    df['coverage_status'] = coverage_statuses
    
    return df