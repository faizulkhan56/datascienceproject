import joblib
from pathlib import Path
from src.datascience.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager().get_model_trainer_config()
        model_path = Path(config.root_dir) / config.model_name
        self.model = joblib.load(model_path)

    def predict(self,data):
        prediction=self.model.predict(data)
        return prediction
