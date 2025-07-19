import json
import mlflow
import logging
import os

# ─── DagsHub MLflow Auth (Hardcoded for dev ONLY) ──────────────────────────
DAGSHUB_USERNAME = "aftabalam1210"
DAGSHUB_TOKEN = "24d2e8f2997ac17d79d399e508358423b4943acf"  # 🔁 replace this

# Set MLflow access credentials for authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Set MLflow Tracking URI linked to your DagsHub repo
mlflow.set_tracking_uri("https://dagshub.com/aftabalam1210/mini-mlops-project.mlflow")

# ─── Logging Setup ─────────────────────────────────────────────────────────
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─── Load Model Info ───────────────────────────────────────────────────────
def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('✅ Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('❌ File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('❌ Unexpected error loading model info: %s', e)
        raise

# ─── Register Model ────────────────────────────────────────────────────────
def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register model into MLflow Model Registry
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition model to "Staging"
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(f"✅ Model '{model_name}' version {model_version.version} registered and promoted to Staging.")
    except Exception as e:
        logger.error('❌ Error during model registration: %s', e)
        raise

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"  # Use a descriptive name
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('❌ Failed to complete model registration: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
