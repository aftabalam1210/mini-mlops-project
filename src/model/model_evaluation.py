import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ───────────────────────────────────────────────────────────────────
# 🔐 MLflow + DagsHub Authentication using Token (No ENV Variables)
# ───────────────────────────────────────────────────────────────────

import mlflow
import os

from dotenv import load_dotenv

load_dotenv() 

# Now safely extract them for your MLflow/DagsHub auth
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Set MLflow env variables securely
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# ✅ Set the MLflow tracking URI to your DagsHub repo
mlflow.set_tracking_uri("https://dagshub.com/aftabalam1210/mini-mlops-project.mlflow")




# ─────────────────────────────────────────────
# 📜 Set up basic logging
# ─────────────────────────────────────────────

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─────────────────────────────────────────────
# 🔧 Utility Functions
# ─────────────────────────────────────────────

def load_model(path: str):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"✅ Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model from {path}: {e}")
        raise

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.debug(f"✅ Data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data from {path}: {e}")
        raise

def evaluate_model(model, X_test, y_test) -> dict:
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        }

        logger.debug("✅ Model evaluation complete")
        return metrics
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}")
        raise

def save_json(data: dict, path: str):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"✅ Saved JSON to {path}")
    except Exception as e:
        logger.error(f"❌ Error saving JSON to {path}: {e}")
        raise

# ─────────────────────────────────────────────
# 🚀 Main Evaluation and Logging Function
# ─────────────────────────────────────────────

def main():
    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:
        try:
            # Load model and data
            model = load_model("models/model.pkl")
            test_df = load_data("data/processed/test_bow.csv")

            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics to MLflow
            for key, val in metrics.items():
                mlflow.log_metric(key, val)

            # Save metrics to file and log
            save_json(metrics, "reports/metrics.json")
            mlflow.log_artifact("reports/metrics.json")

            # Log model parameters
            if hasattr(model, "get_params"):
                for key, val in model.get_params().items():
                    mlflow.log_param(key, val)

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Save model info and log it
            model_info = {"run_id": run.info.run_id, "model_path": "model"}
            # save_json(model_info, "reports/model_info.json")
            # mlflow.log_artifact("reports/model_info.json")
            save_json(model_info, "reports/experiment_info.json")
            mlflow.log_artifact("reports/experiment_info.json")


            # Log error log file to MLflow
            mlflow.log_artifact("model_evaluation_errors.log")

            logger.info("🎉 Model evaluation and logging completed successfully!")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            print(f"❌ Error: {e}")

# ─────────────────────────────────────────────
# 🟩 Run the script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
