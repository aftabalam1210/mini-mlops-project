# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the test class. This method runs once before any tests.
        It handles MLflow setup, authentication, and loading the model and data.
        """
        # --- DagsHub & MLflow Authentication ---
        # Get DagsHub credentials from environment variables set in the CI/CD pipeline
        dagshub_user = os.getenv("DAGSHUB_USERNAME")
        dagshub_token = os.getenv("DAGSHUB_TOKEN")

        # Ensure that the required secrets are available
        if not dagshub_user or not dagshub_token:
            raise EnvironmentError("DAGSHUB_USERNAME and DAGSHUB_TOKEN environment variables must be set.")

        # Set the environment variables that MLflow uses to authenticate with a remote server
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # --- MLflow Tracking URI Configuration ---
        dagshub_url = "https://dagshub.com"
        repo_owner = "aftabalam1210"  # Replace with your DagsHub username if different
        repo_name = "mini-mlops-project" # Replace with your repository name if different
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # --- Model and Data Loading ---
        try:
            # Load the latest "Staging" version of the model from the MLflow Model Registry
            cls.model_name = "my_model"
            latest_version = cls.get_latest_model_version(cls.model_name, stage="Staging")
            if not latest_version:
                raise RuntimeError(f"No model named '{cls.model_name}' found in the 'Staging' stage.")
            
            cls.model_uri = f'models:/{cls.model_name}/{latest_version}'
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)

            # Load the vectorizer used during training
            with open('models/vectorizer.pkl', 'rb') as f:
                cls.vectorizer = pickle.load(f)

            # Load the holdout test data
            cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

        except Exception as e:
            # Provide a more informative error message if setup fails
            raise RuntimeError(f"Failed to set up test class: {e}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        """
        Fetches the latest version number of a model from a specific stage in the MLflow Model Registry.
        """
        client = mlflow.tracking.MlflowClient()
        try:
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            return latest_versions[0].version if latest_versions else None
        except mlflow.exceptions.RestException:
            # Handle case where the model does not exist
            return None

    def test_model_loaded_properly(self):
        """
        Tests if the model object was loaded successfully and is not None.
        """
        self.assertIsNotNone(self.model, "Model should not be None after loading.")

    def test_model_signature(self):
        """
        Tests that the model can make predictions on input with the correct shape and column names.
        This verifies that the model's input signature is as expected.
        """
        # Create a dummy input text
        input_text = ["this is a sample text for prediction"]
        
        # Transform the text using the loaded vectorizer
        input_vector = self.vectorizer.transform(input_text)
        
        # Get feature names from the vectorizer to use as DataFrame columns
        feature_names = self.vectorizer.get_feature_names_out()

        # Create a pandas DataFrame with the correct column names, as expected by the model
        input_df = pd.DataFrame(input_vector.toarray(), columns=feature_names)

        # Predict using the model
        try:
            prediction = self.model.predict(input_df)
        except Exception as e:
            self.fail(f"Model prediction failed with error: {e}")

        # Verify the input shape matches the number of features
        self.assertEqual(input_df.shape[1], len(feature_names), "Input DataFrame column count should match vectorizer feature count.")

        # Verify the output shape (should be one prediction per input row)
        self.assertEqual(len(prediction), input_df.shape[0], "Prediction output should have one entry per input row.")
        self.assertEqual(len(prediction.shape), 1, "Prediction should be a 1D array for binary classification.")

    def test_model_performance(self):
        """
        Tests the model's performance on the holdout dataset against predefined thresholds.
        """
        # Separate features (X) and the target label (y) from the holdout data
        # Assumes the last column is the target
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict on the holdout data
        y_pred = self.model.predict(X_holdout)

        # Calculate performance metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, zero_division=0)
        recall = recall_score(y_holdout, y_pred, zero_division=0)
        f1 = f1_score(y_holdout, y_pred, zero_division=0)
        
        print(f"\n--- Model Performance Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"---------------------------------")

        # Define minimum acceptable performance thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the model's performance meets or exceeds the thresholds
        self.assertGreaterEqual(accuracy, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main(verbosity=2)
