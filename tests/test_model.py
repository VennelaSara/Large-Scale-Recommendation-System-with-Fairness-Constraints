# tests/test_model.py
import unittest
import pandas as pd
from src.features.feature_engineering import generate_features
from src.models.train_model import train_model

class TestRecommendationEngine(unittest.TestCase):
    
    def test_features_generation(self):
        df, ufeat, ifeat = generate_features()
        self.assertTrue("rating_norm" in df.columns)
        self.assertTrue(len(ufeat) > 0)
        self.assertTrue(len(ifeat) > 0)
    
    def test_model_training(self):
        try:
            train_model()
        except Exception as e:
            self.fail(f"Model training failed: {e}")

if __name__ == "__main__":
    unittest.main()
