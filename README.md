# End-to-End Data Science Project

This project is an end-to-end machine learning workflow for the red wine quality dataset. It covers:

- data ingestion
- data validation
- train/test split
- model training with `ElasticNet`
- model evaluation
- experiment tracking with DagsHub-hosted MLflow
- model registration in MLflow Model Registry
- a Flask web app for training and prediction
- containerized execution with Docker and Docker Compose

The repository is structured so that configuration, pipeline orchestration, component logic, tracking, artifacts, and serving are separated cleanly.

## 1. Project Goal

The goal of the project is to take a tabular dataset from source to a trained and tracked model in a reproducible way.

At a high level, the project does this:

1. Download the dataset
2. Validate the schema
3. Split the data into train and test sets
4. Train an `ElasticNet` model
5. Evaluate the model
6. Log metrics and parameters to MLflow on DagsHub
7. Register the model in MLflow as `ElasticnetModel`
8. Serve a simple UI for training and prediction

## 2. End-to-End Flow

The execution flow is:

```text
config.yaml + params.yaml + schema.yaml
            |
            v
ConfigurationManager
            |
            v
Pipeline classes in src/datascience/pipeline/
            |
            v
Component classes in src/datascience/components/
            |
            v
data/ + artifacts/ + logs/ + MLflow/DagsHub
            |
            v
Flask app and prediction UI
```

This separation is important:

- configuration files describe what to run
- entity classes define the structure of that configuration
- component classes do the real work
- pipeline classes orchestrate component execution
- `main.py` runs the whole training workflow
- `app.py` exposes browser routes for training and prediction

## 3. Repository Structure

```text
datascienceproject/
|-- app.py
|-- main.py
|-- data/
|   |-- raw.dvc
|   |-- processed.dvc
|   |-- raw/
|   `-- processed/
|-- .dvc/
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- .env.example
|-- requirements.txt
|-- pyproject.toml
|-- params.yaml
|-- schema.yaml
|-- config/
|   `-- config.yaml
|-- templates/
|   |-- index.html
|   `-- results.html
|-- research/
|   `-- notebooks for stage-by-stage experimentation
`-- src/
    `-- datascience/
        |-- __init__.py
        |-- components/
        |-- config/
        |-- constants/
        |-- entity/
        |-- pipeline/
        `-- utils/
```

## 4. How the Sections Relate to Each Other

### 4.1 `config/config.yaml`

This is the main runtime configuration file.

It defines:

- where DVC-tracked raw data goes
- where DVC-tracked processed data goes
- where the trained model is saved
- where evaluation output is saved
- which MLflow tracking URI to use
- which registered model name to use

Current model registration name:

```yaml
registered_model_name: ElasticnetModel
```

### 4.2 `params.yaml`

This file stores model hyperparameters.

Current parameters:

```yaml
ElasticNet:
  alpha: 0.2
  l1_ratio: 0.1
```

These are read during model training and evaluation.

### 4.3 `schema.yaml`

This file defines the expected dataset columns and target column.

It is used by the validation and training flow to make sure the pipeline is operating on the expected data structure.

Current target column:

```yaml
TARGET_COLUMN:
  name: quality
```

### 4.4 `src/datascience/entity/config_entity.py`

This file defines dataclasses such as:

- `DataIngestionConfig`
- `DataValidationConfig`
- `DataTransformationConfig`
- `ModelTrainerConfig`
- `ModelEvaluationConfig`

These classes turn raw YAML values into typed Python objects.

### 4.5 `src/datascience/config/configuration.py`

This is the configuration manager.

It reads:

- `config.yaml`
- `params.yaml`
- `schema.yaml`

Then it creates stage-specific config objects for every pipeline stage.

### 4.6 `src/datascience/components/`

This folder contains the actual business logic.

Current implemented components:

- `data_ingestion.py`
  Downloads the wine quality ZIP file and extracts it.
- `data_validation.py`
  Checks whether dataset columns match the expected schema.
- `data_transformation.py`
  Splits the dataset into train and test CSV files.
- `model_trainer.py`
  Trains an `ElasticNet` model and saves `model.joblib`.
- `model_evaluation.py`
  Loads the trained model, evaluates it, saves `metrics.json`, logs metrics to MLflow, and registers the model on DagsHub MLflow.

### 4.7 `src/datascience/pipeline/`

This folder wraps each component into a pipeline runner.

It keeps orchestration separate from logic.

Examples:

- `data_ingestion_pipeline.py`
- `model_trainer_pipeline.py`
- `model_evaluation_pipeline.py`
- `prediction_pipeline.py`

### 4.8 `main.py`

This is the end-to-end training entrypoint.

It runs all stages in order:

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Trainer
5. Model Evaluation

If you want to run the full ML pipeline from code or terminal, `main.py` is the main file.

### 4.9 `app.py`

This is the Flask application.

Routes:

- `/`
  Loads the home page
- `/train`
  Starts training in the background
- `/train/status`
  Shows whether background training is running or idle
- `/predict`
  Accepts form values and returns a model prediction

The important design point is that `/train` does not block the web request anymore. It starts `main.py` in a background process and returns immediately.

### 4.10 `artifacts/`

This folder is generated by the pipeline.

Typical contents:

- trained model file
- evaluation metrics JSON

Raw and processed datasets now live under the DVC-tracked `data/` directory:

- `data/raw/`
- `data/processed/`

### 4.11 `logs/`

This folder stores runtime logs.

Important files:

- `logs/logging.log`
  Main application logs
- `logs/training.log`
  Background training log when `/train` is used

## 5. Model and Tracking Behavior

This project trains an `ElasticNet` regressor on wine quality data.

During evaluation, the project logs:

- parameters from `params.yaml`
- metrics such as `rmse`, `mae`, and `r2`
- the trained model artifact
- a registered model in DagsHub-hosted MLflow

The registered model name is:

`ElasticnetModel`

The evaluation flow also tags the registered model version with metric values so the model registry is more informative.

## 6. Prerequisites

Before running the project, make sure you have:

- Python 3.11 or compatible
- Git
- Docker Desktop and Docker Compose
- DVC CLI available locally (installed by `pip install -r requirements.txt`)
- a DagsHub account
- access to the DagsHub repository used for MLflow logging

## 7. Local Setup

### 7.1 Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 7.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 7.3 Create the local `.env`

Create a file named `.env` in the project root.

Example:

```env
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
```

This file is ignored by Git and should not be committed.

### 7.4 DVC note

This repository now tracks dataset directories with DVC:

- `data/raw.dvc`
- `data/processed.dvc`

If your DVC remote is already configured, pull the tracked data with:

```bash
dvc pull
```

If no DVC remote is configured yet, the pipeline can still recreate the local dataset by running `python main.py`, but those files will only remain local until you configure a remote and push them.

## 8. DagsHub and MLflow Setup

### 8.1 Connect the current Git project to DagsHub

You have two common options.

Option 1: connect an existing GitHub repository from the DagsHub UI.

- Open DagsHub
- Create a new repository or connect an existing one
- Follow the "Connect an Existing Repository" flow

Reference:

- DagsHub docs: https://dagshub.com/docs/quick_start/connect_existing_project/

Option 2: add DagsHub as a Git remote to the current repository.

Example:

```bash
git remote add dagshub https://dagshub.com/<username>/<repo-name>.git
git push dagshub main
```

If the remote already exists, inspect it first:

```bash
git remote -v
```

For this project, the MLflow URI currently points to:

`https://dagshub.com/faizulkhan56/datascienceproject.mlflow`

So the matching DagsHub Git repository is typically:

`https://dagshub.com/faizulkhan56/datascienceproject.git`

### 8.2 How MLflow is configured in this project

The MLflow tracking URI is set in [config/config.yaml](/abs/path/C:/faizul-personal/krish-naik-mlops/datascience_project_full/data_science_full_custom/datascienceproject/config/config.yaml:25).

The application loads credentials from `.env`, then `model_evaluation.py` uses them when sending experiment data to DagsHub-hosted MLflow.

### 8.3 How to find `MLFLOW_TRACKING_PASSWORD`

Use a DagsHub access token.

The DagsHub docs recommend:

- `MLFLOW_TRACKING_USERNAME` = your DagsHub username
- `MLFLOW_TRACKING_PASSWORD` = your DagsHub password, or preferably an access token

Reference:

- Experiment tracking docs: https://dagshub.com/docs/feature_guide/experiment_tracking/

In practice, use an access token instead of your account password.

Typical flow:

1. Sign in to DagsHub
2. Open user settings
3. Open the tokens page
4. Create or copy an access token
5. Put that token into `.env` as `MLFLOW_TRACKING_PASSWORD`

The token settings page is:

`https://dagshub.com/user/settings/tokens`

You must be signed in to access it.

### 8.4 How to verify the DagsHub connection

After training completes successfully, you should see:

- a new experiment run in DagsHub MLflow
- a registered model named `ElasticnetModel`
- a new model version
- metrics such as `rmse`, `mae`, and `r2`

### 8.5 DVC and DagsHub data versioning

Experiment tracking and DVC data versioning are separate.

This project now stores datasets under DVC-tracked paths:

- `data/raw/`
- `data/processed/`

To make those datasets visible in DagsHub's data area, you still need to do the account-specific remote setup on your side:

1. Configure a DVC remote that points to your chosen storage backend or DagsHub-supported storage workflow.
2. Authenticate that remote from your machine.
3. Run `dvc push`.
4. Commit and push the generated DVC metadata files.

After that, collaborators can use `dvc pull` after cloning the repo.

## 9. Run the Project Locally Without Docker

### 9.1 Run the full training pipeline

```bash
python main.py
```

What this does:

- downloads or reuses the dataset
- stores raw data under `data/raw/`
- validates columns
- writes train/test splits to `data/processed/`
- trains the model
- evaluates the model
- writes model and metrics artifacts
- logs to DagsHub MLflow

### 9.2 Start the Flask app

```bash
python app.py
```

Then open:

`http://localhost:8080/`

### 9.3 Browser routes

- `http://localhost:8080/`
  Home page
- `http://localhost:8080/train`
  Starts training in the background
- `http://localhost:8080/train/status`
  Returns `running` or `idle`

## 10. Run the Project with Docker

### 10.1 Why Docker is included

Docker makes the project easier to run in a consistent environment.

It packages:

- Python runtime
- project code
- dependencies
- Gunicorn

Docker Compose adds:

- port mapping
- environment file loading
- mounted `data/` for DVC-tracked datasets
- mounted artifacts and logs
- healthcheck

### 10.2 Docker files in this project

#### `Dockerfile`

Purpose:

- builds the runtime image
- installs system and Python dependencies
- copies the project into `/app`
- exposes port `8080`
- runs the Flask app with Gunicorn

#### `docker-compose.yml`

Purpose:

- builds the image from the local `Dockerfile`
- loads values from `.env`
- maps `${APP_HOST_PORT:-8080}` on the host to container port `8080`
- mounts `data/`
- mounts `artifacts/` and `logs/`
- defines a container healthcheck

#### `.dockerignore`

Purpose:

- keeps the build context smaller
- excludes local virtual environments
- excludes local DVC-tracked data payloads
- excludes local logs and artifacts
- excludes `.env`

### 10.3 Build and start with Docker Compose

```bash
docker compose build --no-cache
docker compose up -d
```

### 10.4 Check container status

```bash
docker compose ps
```

Expected state:

- service `web`
- status `Up`
- ideally `(healthy)`

### 10.5 Follow container logs

```bash
docker compose logs -f web
```

### 10.6 Browser routes when running in Docker

- `http://localhost:8080/`
- `http://localhost:8080/train`
- `http://localhost:8080/train/status`

If port `8080` is already used on your machine, override it:

PowerShell:

```powershell
$env:APP_HOST_PORT="8082"
docker compose up -d
```

Git Bash:

```bash
APP_HOST_PORT=8082 docker compose up -d
```

Then use:

`http://localhost:8082/`

### 10.7 Stop Docker services

```bash
docker compose down
```

## 11. How to Use the Project in the Correct Sequence

Recommended sequence for a fresh user:

1. Clone the repository
2. Create `.env`
3. Install dependencies or use Docker
4. Configure DVC remote access if you want shared dataset versioning on DagsHub
5. Check `config/config.yaml`
6. Run the training flow
7. Confirm `data/` and `artifacts/` are generated
8. Confirm MLflow logs appear in DagsHub
9. Use the browser app for prediction and training

## 12. How to Check Results

### 12.1 Local artifact checks

After a successful training run, verify:

- `data/raw/data.zip`
- `data/raw/winequality-red.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `artifacts/model_trainer/model.joblib`
- `artifacts/model_evaluation/metrics.json`
- `data/raw.dvc`
- `data/processed.dvc`

### 12.2 Log checks

Main logs:

```bash
Get-Content .\logs\logging.log -Wait
```

Background training logs:

```bash
Get-Content .\logs\training.log -Wait
```

Quick tail:

```bash
Get-Content .\logs\training.log -Tail 50
```

### 12.3 DagsHub / MLflow checks

After the run, the training logs usually print:

- the MLflow run URL
- the experiment URL

You can also open the repository on DagsHub and then:

1. Open the project page
2. Open the MLflow or Experiments area
3. Open the latest run
4. Inspect metrics, parameters, and artifacts
5. Open the registered model
6. Open the latest model version

What to verify on the run page:

- `rmse`
- `mae`
- `r2`
- run parameters from `params.yaml`
- model artifact

What to verify in the model registry:

- registered model `ElasticnetModel`
- latest version number
- model version created from the run

## 13. How to Find Results from DagsHub

The usual flow is:

1. Open your DagsHub repository
2. Navigate to the experiment tracking / MLflow area
3. Open the latest run
4. Review:
   - metrics
   - parameters
   - artifacts
   - source run information
5. Open the model registry entry
6. Open the latest version of `ElasticnetModel`

If the training log prints a direct link such as:

```text
View run https://dagshub.com/<user>/<repo>.mlflow/#/experiments/0/runs/<run-id>
```

you can open that link directly.

## 14. How to Test the Project

Testing this project is mostly operational and integration-oriented.

### 14.1 Basic static checks

Syntax check:

```bash
python -m py_compile app.py
python -m py_compile main.py
```

### 14.2 Local pipeline check

```bash
python main.py
```

Success criteria:

- no exception raised
- `data/raw/` and `data/processed/` created or reused
- metrics JSON written
- model saved
- DagsHub MLflow run created

### 14.3 Web app check

Start app:

```bash
python app.py
```

Then verify:

- `/` loads
- `/train` starts background training
- `/train/status` changes to `running` and later returns to `idle`
- `/predict` returns a prediction when the form is submitted

### 14.4 Docker check

```bash
docker compose build --no-cache
docker compose up -d
docker compose ps
docker compose logs -f web
```

Success criteria:

- container starts
- healthcheck passes
- Gunicorn is running
- `/train` works without blocking the browser

## 15. Common Operational Checks

### Check whether training is running

Browser:

`http://localhost:8080/train/status`

Expected:

- `running` during active training
- `idle` after training completes

### Check whether training completed successfully

Look for the final stage completion lines in `logs/training.log`, especially:

- `Model Evaluation stage completed`
- the MLflow run URL
- the registered model version creation line

### Check whether the model exists locally

```bash
dir artifacts\model_trainer
```

Expected:

- `model.joblib`

### Check whether DVC-tracked data exists locally

```bash
dir data\raw
dir data\processed
```

Optional DVC check:

```bash
dvc status
```

### Check whether metrics exist locally

```bash
type artifacts\model_evaluation\metrics.json
```

## 16. Troubleshooting

### Problem: browser `/train` hangs or shows an error page

Cause:

- old behavior blocked the web request while training ran

Current solution:

- `/train` now starts background training with `subprocess.Popen`

Action:

- rebuild Docker if needed
- call `/train`
- monitor `logs/training.log`

### Problem: Gunicorn worker timeout

Cause:

- stale image or old server configuration
- connection-level stalls

Action:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Problem: no MLflow logs appear on DagsHub

Check:

- `.env` exists
- `MLFLOW_TRACKING_USERNAME` is correct
- `MLFLOW_TRACKING_PASSWORD` contains a valid DagsHub token
- `config/config.yaml` has the correct `mlflow_uri`

### Problem: model version exists but metrics are not obvious on the model version page

Understand the separation:

- metrics are primarily logged to the MLflow run
- the registered model links to the run
- this project also tags model versions with metrics, but the run page is still the main place to inspect experiment metrics

### Problem: `uv` warning in MLflow logs

This is not fatal.

MLflow tries to infer environment metadata using `uv`, then falls back automatically if `uv` is unavailable.

### Problem: Git warning from MLflow inside Docker

The Docker image now installs `git`, and `GIT_PYTHON_REFRESH=quiet` is set to reduce noisy warnings.

## 17. What a Successful Run Looks Like

A successful run usually includes:

- all five pipeline stages completing
- raw data under `data/raw/`
- processed data under `data/processed/`
- `metrics.json` being saved
- a new MLflow run URL in the logs
- a new `ElasticnetModel` version created
- `/train/status` eventually returning `idle`

## 18. Recommended Daily Workflow

For normal development:

1. Pull latest code
2. Update config or parameters if needed
3. Start Docker or local environment
4. Trigger training
5. Monitor logs
6. Check DagsHub MLflow results
7. Test prediction UI
8. Commit only intended changes

## 19. Useful Commands Reference

Install:

```bash
pip install -r requirements.txt
```

Optional DVC pull:

```bash
dvc pull
```

Run pipeline:

```bash
python main.py
```

Run app:

```bash
python app.py
```

Docker build:

```bash
docker compose build --no-cache
```

Docker start:

```bash
docker compose up -d
```

Docker logs:

```bash
docker compose logs -f web
```

Docker stop:

```bash
docker compose down
```

Training log tail:

```bash
Get-Content .\logs\training.log -Wait
```

Training status:

```text
http://localhost:8080/train/status
```

## 20. External References

- DagsHub experiment tracking docs: https://dagshub.com/docs/feature_guide/experiment_tracking/
- DagsHub track ML experiments guide: https://dagshub.com/docs/use_cases/track_ml_experiments/
- DagsHub connect existing repository guide: https://dagshub.com/docs/quick_start/connect_existing_project/
- MLflow docs: https://mlflow.org/docs/latest/

## 21. Final Notes

This repository is now structured so that a new user can:

1. set up credentials
2. run the training flow
3. inspect generated artifacts
4. inspect MLflow results in DagsHub
5. use Docker for a repeatable environment
6. test the browser-based training and prediction flow

If you keep `.env` local, rebuild Docker after runtime changes, and monitor `logs/training.log` plus DagsHub MLflow, the project is straightforward to operate end-to-end.
