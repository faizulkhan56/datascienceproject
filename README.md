# End to End Data Science Project

This custom project follows the same high-level structure as the original project, but it is being rebuilt step by step for practice. The goal is to keep configuration, reusable utilities, pipeline stages, and app entrypoints clearly separated.

## Current Project Structure

```text
datascienceproject/
├── app.py
├── main.py
├── template.py
├── pyproject.toml
├── requirements.txt
├── params.yaml
├── schema.yaml
├── config/
│   └── config.yaml
├── research/
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_validation.ipynb
│   ├── 3_data_transformation.ipynb
│   ├── 4_model_trainer.ipynb
│   ├── 5_model_evaluation.ipynb
│   └── research.ipynb
├── src/
│   └── datascience/
│       ├── __init__.py
│       ├── components/
│       ├── config/
│       │   └── configuration.py
│       ├── constants/
│       │   └── __init__.py
│       ├── entity/
│       │   └── config_entity.py
│       ├── pipeline/
│       └── utils/
│           └── common.py
└── templates/
    └── index.html
```

## Workflows: ML Pipeline

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Trainer
5. Model Evaluation

These stages describe the logical ML flow. In this project, each stage is expected to be implemented in three places:

- `src/datascience/components/`
  Contains the actual stage logic such as downloading data, validating schema, splitting data, training the model, and evaluating metrics.
- `src/datascience/pipeline/`
  Contains stage runner classes that call the component logic in execution order.
- `main.py`
  Orchestrates the complete end-to-end training flow by calling each pipeline stage.

## Workflow-to-File Relationship

### 1. Update `config.yaml`

File:
- [config/config.yaml](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/config/config.yaml)

Purpose:
- Stores project paths and stage-level settings
- Defines where artifacts, datasets, model files, and evaluation outputs should be saved
- Acts as the main runtime configuration file

Used by:
- `src/datascience/config/configuration.py`
- future component and pipeline classes

### 2. Update `schema.yaml`

File:
- [schema.yaml](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/schema.yaml)

Purpose:
- Defines expected dataset columns and target column
- Helps validation logic confirm that incoming data matches the expected structure

Used by:
- `src/datascience/config/configuration.py`
- future data validation component

### 3. Update `params.yaml`

File:
- [params.yaml](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/params.yaml)

Purpose:
- Stores model hyperparameters such as values for training configuration
- Keeps experiment settings outside the code

Used by:
- `src/datascience/config/configuration.py`
- future model training and evaluation components

### 4. Update the Entity

Files:
- [src/datascience/entity/config_entity.py](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/src/datascience/entity/config_entity.py)

Purpose:
- Defines structured config classes using dataclasses
- Converts raw config values into strongly organized Python objects
- Makes stage code cleaner by passing one config object instead of many loose values

Example use:
- `DataIngestionConfig`
- `DataValidationConfig`
- `ModelTrainerConfig`

### 5. Update the Configuration Manager in `src/config`

Files:
- [src/datascience/config/configuration.py](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/src/datascience/config/configuration.py)
- [src/datascience/constants/__init__.py](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/src/datascience/constants/__init__.py)

Purpose:
- Reads `config.yaml`, `schema.yaml`, and `params.yaml`
- Creates stage-specific config objects from the entity layer
- Centralizes configuration loading for the entire project

Why it matters:
- Components should not read raw YAML files directly
- The configuration manager prepares validated, reusable config objects for every stage

### 6. Update the Components

Folder:
- [src/datascience/components](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/src/datascience/components)

Purpose:
- Holds the main business logic for each ML stage

Expected files to create later:
- `data_ingestion.py`
- `data_validation.py`
- `data_transformation.py`
- `model_trainer.py`
- `model_evaluation.py`

Examples of responsibility:
- ingestion downloads and extracts data
- validation checks schema and data quality
- transformation prepares train/test data
- trainer fits the model and saves artifacts
- evaluation calculates metrics and logs results

Current status:
- the folder exists, but the stage files are not implemented yet

### 7. Update the Pipeline

Folder:
- [src/datascience/pipeline](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/src/datascience/pipeline)

Purpose:
- Wraps component execution in pipeline classes
- Keeps orchestration separate from stage logic

Expected files to create later:
- `data_ingestion_pipeline.py`
- `data_validation_pipeline.py`
- `data_transformation_pipeline.py`
- `model_trainer_pipeline.py`
- `model_evaluation_pipeline.py`
- optionally `prediction_pipeline.py`

Current status:
- the folder exists, but the pipeline files are not implemented yet

### 8. Update `main.py`

File:
- [main.py](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/main.py)

Purpose:
- Serves as the main execution entrypoint
- Calls each training stage in sequence
- Uses logging to show pipeline progress

Current status:
- right now it only logs a welcome message and prints a test message
- later it should run the full training workflow

## Supporting Files

### `src/datascience/__init__.py`

Purpose:
- sets up project-level logging
- creates the logger object used across the project

### `src/datascience/utils/common.py`

Purpose:
- stores shared helper functions used across many files
- examples include reading YAML, creating directories, saving JSON, and loading artifacts

Why it exists:
- avoids repeating the same helper code in every component

### `research/`

Purpose:
- notebook-based experimentation area
- useful for trying each ML stage interactively before moving code into `src/`

Suggested use:
- test ideas in notebooks first
- then move stable logic into `components/` and `pipeline/`

### `app.py`

Purpose:
- web app entrypoint for user interaction
- can later expose training and prediction routes

### `templates/`

Purpose:
- HTML templates for the Flask app UI

## Recommended Development Order

1. Finalize `config/config.yaml`
2. Finalize `schema.yaml`
3. Finalize `params.yaml`
4. Complete `config_entity.py`
5. Complete `configuration.py`
6. Implement files in `components/`
7. Implement files in `pipeline/`
8. Upgrade `main.py` to run the full training flow
9. Connect `app.py` to prediction or training routes

## Current State Summary

Implemented now:
- project scaffolding
- logging setup
- utility helpers
- notebook folder
- config/entity/configuration skeleton

Still to implement:
- component stage files
- pipeline stage files
- full training orchestration in `main.py`
- artifact generation flow
- model serving flow in `app.py`
