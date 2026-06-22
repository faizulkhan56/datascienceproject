import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from mlflow.tracking import MlflowClient

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _load_local_env(self):
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def _ensure_tracking_credentials(self):
        tracking_scheme = urlparse(self.config.mlflow_uri).scheme
        if tracking_scheme not in {"http", "https"}:
            return

        if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
            return

        raise ValueError(
            "Remote MLflow tracking requires MLFLOW_TRACKING_USERNAME and "
            "MLFLOW_TRACKING_PASSWORD environment variables."
        )

    def _tag_registered_model_version(self, run_id: str, metrics: dict):
        client = MlflowClient()
        model_versions = client.search_model_versions(
            f"name = '{self.config.registered_model_name}'"
        )

        matching_versions = [
            version for version in model_versions if version.run_id == run_id
        ]
        if not matching_versions:
            return

        latest_version = max(matching_versions, key=lambda version: int(version.version))

        for metric_name, metric_value in metrics.items():
            client.set_model_version_tag(
                name=self.config.registered_model_name,
                version=latest_version.version,
                key=metric_name,
                value=f"{metric_value:.6f}",
            )
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        self._load_local_env()
        self._ensure_tracking_credentials()
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run() as run:

            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=self.config.registered_model_name,
                )
                self._tag_registered_model_version(run.info.run_id, scores)
            else:
                mlflow.sklearn.log_model(model, "model")
