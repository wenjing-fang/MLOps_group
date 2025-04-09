import unittest
from ml import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

class TestModels(unittest.TestCase):

    def setUp(self):
        # Create a small sample of data
        self.X_train = pd.DataFrame({
            'contact': [0, 1],
            'housing': [1, 0],
            'duration': [120, 340],
            'pdays': [999, 5],
            'previous': [0, 2],
        })
        self.y_train = [1, 0]

    def test_get_model_logistic(self):
        model = models.get_model('logistic')
        self.assertIsInstance(model, LogisticRegression)

    def test_get_model_rf(self):
        model = models.get_model('rf')
        self.assertIsInstance(model, RandomForestClassifier)

    def test_get_model_svm(self):
        model = models.get_model('svm')
        self.assertIsInstance(model, SVC)

    def test_get_model_invalid(self):
        with self.assertRaises(ValueError):
            models.get_model('banana')

    def test_train_model(self):
        model = models.get_model('rf')
        trained_model = models.train_model(model, self.X_train, self.y_train)
        self.assertTrue(hasattr(trained_model, "predict"))

if __name__ == '__main__':
    unittest.main()