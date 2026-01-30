# src/ingestion/etl_pipeline.py
import os
import pandas as pd
from pyspark.sql import SparkSession

from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH

def load_raw_data():
    # Example using MovieLens dataset
    users = pd.read_csv(os.path.join(DATA_RAW_PATH, "users.csv"))
    items = pd.read_csv(os.path.join(DATA_RAW_PATH, "movies.csv"))
    interactions = pd.read_csv(os.path.join(DATA_RAW_PATH, "ratings.csv"))
    return users, items, interactions

def preprocess(users, items, interactions):
    # Merge user-item interactions
    df = interactions.merge(users, on="userId", how="left")
    df = df.merge(items, on="movieId", how="left")

    # Example: encode categorical columns
    df["gender"] = df["gender"].map({"M":0,"F":1})
    df["category_encoded"] = df["genres"].astype('category').cat.codes
    df.to_csv(os.path.join(DATA_PROCESSED_PATH, "interactions_processed.csv"), index=False)
    print("Preprocessed data saved.")
    return df

if __name__ == "__main__":
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    users, items, interactions = load_raw_data()
    preprocess(users, items, interactions)
