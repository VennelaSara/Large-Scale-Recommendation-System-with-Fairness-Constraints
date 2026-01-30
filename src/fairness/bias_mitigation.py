# src/fairness/bias_mitigation.py
import pandas as pd

def mitigate_bias(recommendations, user_df, sensitive_col="gender", threshold=0.8):
    """
    Re-rank recommendations to improve demographic parity.
    recommendations: dict {user_id: [item_ids]}
    user_df: DataFrame with userId and sensitive_col
    threshold: minimum acceptable exposure ratio
    """
    # Count exposure per group
    exposure = {}
    for user_id, items in recommendations.items():
        group = user_df.loc[user_df["userId"]==user_id, sensitive_col].values[0]
        exposure[group] = exposure.get(group, 0) + len(items)
    
    # Compute ratio
    min_group = min(exposure.values())
    max_group = max(exposure.values())
    ratio = min_group / max_group if max_group>0 else 1.0
    
    # If below threshold, apply simple balancing
    if ratio < threshold:
        # Shuffle and redistribute items across users of underexposed group
        under_group = min(exposure, key=exposure.get)
        for user_id, items in recommendations.items():
            group = user_df.loc[user_df["userId"]==user_id, sensitive_col].values[0]
            if group == under_group:
                recommendations[user_id] = sorted(items, reverse=True)
    
    return recommendations
