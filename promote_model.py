# promote_model.py
# This script promotes the latest "Staging" model to "Production"
# in the MLflow Model Registry.

import os
import mlflow

def promote_model():
    """
    Finds the latest model version in the 'Staging' stage, archives any
    current 'Production' models, and promotes the 'Staging' model to 'Production'.
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
    # IMPORTANT: Make sure these match your DagsHub repository details
    dagshub_url = "https://dagshub.com"
    repo_owner = "aftabalam1210"
    repo_name = "mini-mlops-project"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    # --- Model Promotion Logic ---
    client = mlflow.tracking.MlflowClient()
    model_name = "my_model"

    # 1. Find the latest model version in the "Staging" stage
    print(f"Checking for the latest version of model '{model_name}' in 'Staging'...")
    latest_staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

    if not latest_staging_versions:
        print("No models found in 'Staging'. Exiting.")
        return # Exit the script if there's nothing to promote

    staging_version_to_promote = latest_staging_versions[0]
    print(f"Found version {staging_version_to_promote.version} to promote.")

    # 2. Archive any existing models in the "Production" stage
    print("Checking for existing models in 'Production' to archive...")
    production_versions = client.get_latest_versions(model_name, stages=["Production"])

    for version in production_versions:
        print(f"Archiving production version {version.version}...")
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # 3. Promote the "Staging" model to "Production"
    print(f"Promoting version {staging_version_to_promote.version} to 'Production'...")
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version_to_promote.version,
        stage="Production"
    )

    print("\n✅ Success!")
    print(f"Model '{model_name}' version {staging_version_to_promote.version} has been promoted to 'Production'.")

if __name__ == "__main__":
    promote_model()
