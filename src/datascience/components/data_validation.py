import pandas as pd

from src.datascience.entity.config_entity import DataValidationConfig


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = set(data.columns)
            all_schema = set(self.config.all_schema.keys())
            validation_status = all_cols == all_schema

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e

    

