from fastapi import FastAPI
from redis_cache import get_cached_recommendations, set_cached_recommendations
import pandas as pd
import tensorflow as tf
from ranking_model import build_two_tower_model
from reranker import rerank
from config import MODEL_DIR, TOP_K

app = FastAPI(title="Recommendation Engine")

# Load model
model = tf.keras.models.load_model(f"{MODEL_DIR}/two_tower_model")
user_features = pd.read_csv("../data/processed/user_features.csv")
item_features = pd.read_csv("../data/processed/item_features.csv")

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    # Check cache
    cached = get_cached_recommendations(user_id)
    if cached:
        return {"user_id": user_id, "recommendations": cached}
    
    # Predict relevance for all items
    item_ids = item_features["movieId"].tolist()
    user_ids = [user_id]*len(item_ids)
    preds = model.predict([user_ids, item_ids], verbose=0)
    candidates = list(zip(item_ids, preds.flatten()))
    
    # Re-rank with multi-objective optimization
    top_items = rerank(user_id, candidates, user_features, item_features)
    recommended_ids = [item for item, score in top_items]
    
    # Set cache
    set_cached_recommendations(user_id, recommended_ids)
    return {"user_id": user_id, "recommendations": recommended_ids}
