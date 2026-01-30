# src/config.py

DATA_RAW_PATH = "./data/raw"
DATA_PROCESSED_PATH = "./data/processed"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

POSTGRES_URI = "postgresql://postgres:postgres@localhost:5432/recommendation_db"

MODEL_DIR = "./models"
EMBEDDING_DIM = 64
TOP_K = 10

# Multi-objective weights
WEIGHT_RELEVANCE = 0.6
WEIGHT_DIVERSITY = 0.2
WEIGHT_REVENUE = 0.2

# Fairness threshold
FAIRNESS_THRESHOLD = 0.8

# Batch prediction
BATCH_SIZE = 128
