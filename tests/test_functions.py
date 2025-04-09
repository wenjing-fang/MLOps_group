import unittest
import pandas as pd
from ml import functions

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Simulate a small version of your dataset
        self.raw_data = pd.DataFrame({
            'contact': ['cellular', 'telephone'],
            'housing': ['yes', 'no'],
            'duration': [120, 340],
            'pdays': [999, 5],
            'previous': [0, 2],
            'deposit': ['yes', 'no']
        })

    def test_preprocess_data(self):
        df_processed = functions.preprocess_data(self.raw_data)
        self.assertIn('deposit', df_processed.columns) #
        self.assertTrue(df_processed['deposit'].isin([0, 1]).all())

    def test_preprocess_user_data(self):
        df_user = functions.preprocess_user_data(self.raw_data)
        self.assertNotIn('deposit', df_user.columns)
        self.assertIn('contact', df_user.columns)

    def test_split_data(self):
        df = functions.preprocess_data(self.raw_data)
        X_train, X_test, y_train, y_test = functions.split_data(df)
        self.assertEqual(len(X_train) + len(X_test), len(df))
        self.assertEqual(X_train.shape[1], 5)  # 5 features excluding 'deposit'

    def test_evaluate_model(self):
        from sklearn.ensemble import RandomForestClassifier

        df = functions.preprocess_data(self.raw_data)
        X_train, X_test, y_train, y_test = functions.split_data(df)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        f1, precision = functions.evaluate_model(model, X_test, y_test)

        self.assertIsInstance(f1, float)
        self.assertIsInstance(precision, float)

if __name__ == '__main__':
    unittest.main()