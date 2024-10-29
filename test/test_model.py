import unittest
from model_training import train_model
import pandas as pd

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        # Sample data for testing
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.5, 1.5, 2.5, 3.5],
            'target': [1, 0, 1, 0]
        })
        
        model = train_model(data)
        self.assertIsNotNone(model)  # Check if the model is trained

if __name__ == '__main__':
    unittest.main()