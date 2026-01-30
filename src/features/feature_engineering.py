import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_PROCESSED_PATH

def generate_features():
    df = pd.read_csv(f"{DATA_PROCESSED_PATH}/interactions_processed.csv")
    
    # User features: number of interactions
    user_features = df.groupby("userId").size().reset_index(name="num_interactions")
    
    # Item features: popularity & revenue (simulated)
    item_features = df.groupby("movieId").size().reset_index(name="popularity")
    item_features["revenue"] = np.random.uniform(0, 100, size=len(item_features))
    
    # Interaction features: normalized rating
    df["rating_norm"] = MinMaxScaler().fit_transform(df[["rating"]])
    
    user_features.to_csv(f"{DATA_PROCESSED_PATH}/user_features.csv", index=False)
    item_features.to_csv(f"{DATA_PROCESSED_PATH}/item_features.csv", index=False)
    df.to_csv(f"{DATA_PROCESSED_PATH}/interactions_features.csv", index=False)
    print("Feature generation complete.")
    return df, user_features, item_features

if __name__ == "__main__":
    generate_features()
