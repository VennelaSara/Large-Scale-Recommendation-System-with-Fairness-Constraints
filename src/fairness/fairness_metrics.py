import pandas as pd

def demographic_parity(recommendations, user_df, sensitive_col="gender"):
    """
    recommendations: dict {user_id: [item_ids]}
    user_df: DataFrame with userId and sensitive_col
    Returns: ratio of exposure across groups
    """
    exposure = {}
    for user_id, items in recommendations.items():
        group = user_df.loc[user_df["userId"]==user_id, sensitive_col].values[0]
        exposure[group] = exposure.get(group, 0) + len(items)
    
    values = list(exposure.values())
    ratio = min(values)/max(values) if max(values) > 0 else 1
    return ratio
