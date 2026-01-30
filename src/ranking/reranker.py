import numpy as np
from src.config import WEIGHT_RELEVANCE, WEIGHT_DIVERSITY, WEIGHT_REVENUE

def rerank(user_id, candidates, user_features, item_features):
    """
    candidates: list of item_ids with predicted relevance scores
    item_features: DataFrame with popularity & revenue
    """
    final_scores = []
    for item_id, relevance in candidates:
        diversity_score = 1.0 / (1.0 + item_features.loc[item_features["movieId"]==item_id, "popularity"].values[0])
        revenue_score = item_features.loc[item_features["movieId"]==item_id, "revenue"].values[0] / 100.0
        
        final_score = (
            WEIGHT_RELEVANCE*relevance +
            WEIGHT_DIVERSITY*diversity_score +
            WEIGHT_REVENUE*revenue_score
        )
        final_scores.append((item_id, final_score))
    
    # Sort by final score descending
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores[:10]  # top 10
