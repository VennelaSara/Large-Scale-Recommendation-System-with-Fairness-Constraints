import redis
import pickle
from config import REDIS_HOST, REDIS_PORT, REDIS_DB

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def get_cached_recommendations(user_id):
    cached = r.get(f"user:{user_id}:recommendations")
    if cached:
        return pickle.loads(cached)
    return None

def set_cached_recommendations(user_id, recommendations):
    r.set(f"user:{user_id}:recommendations", pickle.dumps(recommendations), ex=3600)  # 1 hour TTL
