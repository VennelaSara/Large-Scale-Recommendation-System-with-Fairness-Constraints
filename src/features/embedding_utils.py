# src/features/embedding_utils.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def create_user_item_embeddings(df, user_col="userId", item_col="movieId", embedding_dim=64):
    """
    Generate user and item embeddings using one-hot encoding and Dense layer.
    Useful for initializing embedding matrices.
    """
    # Encode users/items as integer indices
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df[user_col] = user_encoder.fit_transform(df[user_col])
    df[item_col] = item_encoder.fit_transform(df[item_col])
    
    num_users = df[user_col].nunique()
    num_items = df[item_col].nunique()
    
    # Initialize random embeddings
    user_embedding_matrix = np.random.rand(num_users, embedding_dim).astype(np.float32)
    item_embedding_matrix = np.random.rand(num_items, embedding_dim).astype(np.float32)
    
    return user_embedding_matrix, item_embedding_matrix, user_encoder, item_encoder
