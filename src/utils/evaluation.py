# src/utils/evaluation.py
import numpy as np

def ndcg_at_k(y_true, y_score, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at K
    """
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    gains = 2**y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    return np.sum(gains / discounts)

def precision_at_k(y_true, y_score, k=10):
    """
    Precision at K
    """
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    return np.sum(y_true_sorted > 0) / k
