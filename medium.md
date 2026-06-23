# End-to-End Data Science Project — Complete Guide

This document is the **one-stop guideline** for this repository. It combines:

- practical operations from `README.md`
- theoretical background on the dataset and algorithm
- explanation of how every part of the codebase connects
- guidance on metrics, tracking, and improving wine quality predictions

Use this file when you want to understand **what** the project does, **why** it is designed this way, and **how** to run and improve it.

---

# Part I — Conceptual Foundation

## 1. What This Project Is

This project is an end-to-end machine learning workflow for the **red wine quality dataset**. It covers:

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

### 1.1 Project Goal

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

---

## 2. Wine Quality Dataset — Theory and Context

### 2.1 Origin and Purpose

The **Wine Quality** dataset is a well-known tabular dataset used in machine learning education and benchmarking. It was originally published in connection with wine chemistry research and is commonly used to predict **sensory quality scores** from measurable chemical properties.

In this project we use the **red wine** subset. The data describes physicochemical measurements of individual wine samples. Each row is one wine sample. The target variable is a **quality score** assigned by human tasters (not a continuous chemical measurement).

### 2.2 Problem Type

This is a **regression** problem:

- **Input (features):** 11 numeric wine chemistry attributes
- **Output (target):** `quality` — an integer score, typically between 3 and 8 in practice

The model learns a mapping:

```text
(fixed acidity, volatile acidity, ..., alcohol)  -->  predicted quality
```

### 2.3 Feature Descriptions

| Feature | Meaning | Role in Wine Chemistry |
|---------|---------|------------------------|
| **fixed acidity** | Tartaric acid content | Contributes to sharpness and stability |
| **volatile acidity** | Acetic acid content | High values often indicate spoilage or vinegar-like taste |
| **citric acid** | Citric acid content | Can add freshness; usually low in red wines |
| **residual sugar** | Sugar remaining after fermentation | Affects sweetness perception |
| **chlorides** | Salt content | Influences taste balance |
| **free sulfur dioxide** | Unbound SO₂ | Preservative; affects oxidation control |
| **total sulfur dioxide** | Total SO₂ (free + bound) | Preservation and microbial control |
| **density** | Wine density | Related to sugar and alcohol content |
| **pH** | Acidity level on pH scale | Lower pH = more acidic wine |
| **sulphates** | Potassium sulphate additive | Can affect SO₂ levels and preservation |
| **alcohol** | Alcohol by volume (%) | Strong influence on body and perceived quality |

### 2.4 Target Column: `quality`

- Defined in `schema.yaml` as `quality` (integer)
- Represents an **aggregated expert rating**, not a precise chemical measurement
- Because ratings are subjective and discrete, perfect prediction is difficult
- Small prediction errors (e.g., predicting 5.8 vs actual 6) may still be acceptable in practice

### 2.5 Dataset Characteristics Relevant to Modeling

Understanding the data helps explain model behavior:

1. **Moderate size** — roughly 1,600 red wine samples; enough for baseline models but not deep learning scale
2. **All numeric features** — no categorical encoding required in the current pipeline
3. **No scaling in current pipeline** — features have different ranges (e.g., density ~0.99 vs free SO₂ ~0–70); scaling can improve linear models
4. **Imbalanced target distribution** — most wines cluster around scores 5–6; rare scores (3, 8) are harder to predict
5. **Correlated features** — density, alcohol, and sugar-related features often correlate; regularized models like ElasticNet handle this better than plain linear regression

### 2.6 Where the Dataset Lives in This Project

| Stage | Location |
|-------|----------|
| Remote source | GitHub ZIP URL in `config/config.yaml` |
| Downloaded archive | `data/raw/data.zip` |
| Extracted CSV | `data/raw/winequality-red.csv` |
| Train split | `data/processed/train.csv` |
| Test split | `data/processed/test.csv` |

---

## 3. ElasticNet Algorithm — Theory

### 3.1 What ElasticNet Is

**ElasticNet** is a linear regression model that combines:

- **L1 regularization (Lasso)** — encourages sparsity (some coefficients become exactly zero)
- **L2 regularization (Ridge)** — shrinks coefficients smoothly and handles multicollinearity

It is implemented in scikit-learn as `sklearn.linear_model.ElasticNet`.

### 3.2 Why ElasticNet Fits This Project

For wine quality prediction, ElasticNet is a strong baseline because:

1. The relationship between chemistry and perceived quality is often **approximately linear** at first approximation
2. Features are **correlated** (e.g., density and alcohol)
3. The model is **interpretable** — you can inspect which chemical factors push quality up or down
4. It is **fast**, **stable**, and easy to deploy with `joblib`
5. Hyperparameters are simple to tune and log in MLflow

### 3.3 Mathematical Intuition

ElasticNet minimizes:

```text
Loss = MSE + α × [ l1_ratio × |w| + (1 - l1_ratio) × w² ]
```

Where:

- **MSE** — mean squared error between predicted and actual quality
- **w** — model coefficients (weights for each feature)
- **α (alpha)** — overall regularization strength
- **l1_ratio** — balance between Lasso (1.0) and Ridge (0.0)

### 3.4 Hyperparameters in This Project

Defined in `params.yaml`:

```yaml
ElasticNet:
  alpha: 0.2
  l1_ratio: 0.1
```

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `alpha` | 0.2 | Higher → stronger penalty → simpler model, may underfit if too high |
| `l1_ratio` | 0.1 | Closer to 0 → more Ridge-like; closer to 1 → more Lasso-like (feature selection) |

These values are read by `ConfigurationManager`, passed into `ModelTrainerConfig`, used during training in `model_trainer.py`, and logged to MLflow during evaluation.

### 3.5 Training Behavior in This Codebase

In `model_trainer.py`:

1. Load `train.csv` and `test.csv`
2. Split features (`X`) and target (`y = quality`)
3. Create `ElasticNet(alpha=..., l1_ratio=..., random_state=42)`
4. Fit on training data only
5. Save fitted model to `artifacts/model_trainer/model.joblib`

At prediction time (`prediction_pipeline.py`), the saved model loads and runs `model.predict()` on user input from the Flask form.

---

## 4. Evaluation Metrics — Theory

During model evaluation, the project computes and logs three regression metrics.

### 4.1 RMSE (Root Mean Squared Error)

**Formula:**

```text
RMSE = sqrt( mean( (y_true - y_pred)² ) )
```

**Interpretation:**

- Measures average prediction error in the **same units as quality score**
- Penalizes large errors more heavily than MAE (because errors are squared)
- Lower is better
- Example: RMSE ≈ 0.70 means predictions are off by roughly 0.7 quality points on average (with extra penalty for big misses)

**In this project:** computed in `model_evaluation.py` via `mean_squared_error`, then square-rooted.

### 4.2 MAE (Mean Absolute Error)

**Formula:**

```text
MAE = mean( |y_true - y_pred| )
```

**Interpretation:**

- Average absolute difference between predicted and actual quality
- Easier to explain to non-technical stakeholders than RMSE
- Less sensitive to occasional large outliers than RMSE
- Lower is better

**In this project:** if MAE ≈ 0.54, predictions are typically within about half a quality point.

### 4.3 R² (Coefficient of Determination)

**Formula (intuition):**

```text
R² = 1 - (variance unexplained by model / total variance in target)
```

**Interpretation:**

- Ranges from negative infinity to 1.0 (1.0 is perfect)
- Measures **how much variance in quality** the model explains
- R² ≈ 0.21 means the model explains about 21% of quality variation — common for this dataset with a simple linear model
- Higher is better

**Important:** for subjective ratings like wine quality, even strong models often show modest R². Improvement should be judged relative to baselines, not against an unrealistic R² = 1.0.

### 4.4 Where Metrics Are Stored and Tracked

| Location | File / System |
|----------|---------------|
| Local JSON | `artifacts/model_evaluation/metrics.json` |
| MLflow run | logged as `rmse`, `mae`, `r2` |
| Model Registry tags | version tags on `ElasticnetModel` in DagsHub |

---

## 5. How to Improve Model Quality and Predictions

Better wine quality predictions and better metrics usually move together, but always validate on the **held-out test set** to avoid overfitting.

### 5.1 Hyperparameter Tuning

Adjust `params.yaml` and re-run `python main.py`:

- Try a grid of `alpha` values: `0.01, 0.05, 0.1, 0.2, 0.5, 1.0`
- Try `l1_ratio` values: `0.1, 0.3, 0.5, 0.7, 0.9`
- Use `GridSearchCV` or `RandomizedSearchCV` in a research notebook
- Log each experiment in MLflow to compare runs

**Expected effect:** lower RMSE/MAE, higher R² if tuning finds a better bias-variance tradeoff.

### 5.2 Feature Scaling

The current pipeline does **not** scale features. ElasticNet is sensitive to feature scale because regularization penalizes all coefficients equally.

**Improvement:** add `StandardScaler` or `MinMaxScaler` in `data_transformation.py` or wrap the model in a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=..., l1_ratio=..., random_state=42)),
])
```

**Expected effect:** often improves linear model performance noticeably.

### 5.3 Better Train/Test Methodology

Current split in `data_transformation.py` uses default `train_test_split(data)` (75/25, shuffled, no `random_state`).

**Improvements:**

- Set `random_state=42` for reproducibility
- Try stratified splitting based on binned quality scores
- Use cross-validation for more reliable metric estimates

### 5.4 Feature Engineering

Ideas grounded in wine chemistry:

- Interaction terms (e.g., alcohol × volatile acidity)
- Ratios (e.g., free SO₂ / total SO₂)
- Polynomial features for non-linear effects
- Outlier handling for extreme volatile acidity values

### 5.5 Try Alternative Models

Keep ElasticNet as baseline, then compare in MLflow:

| Model | When to Try |
|-------|-------------|
| **Ridge / Lasso** | Isolate effect of L1 vs L2 |
| **Random Forest** | Capture non-linear relationships |
| **Gradient Boosting (XGBoost, LightGBM)** | Often strong on tabular data |
| **SVR** | Non-linear with kernel trick |

Register the best model under a new name or version in MLflow Model Registry.

### 5.6 Data Quality Improvements

- Strengthen validation in `data_validation.py` (check dtypes, nulls, ranges — not just column names)
- Detect and handle duplicate rows
- Analyze target distribution and report class imbalance

### 5.7 Connecting Metrics to User-Facing Predictions

When metrics improve:

- `/predict` returns quality estimates closer to actual scores
- Users see more trustworthy results in `results.html`
- MLflow runs show a history of improvement across experiments

**Practical loop:**

```text
change params/features → run main.py → check metrics.json → compare in DagsHub → deploy better model.joblib
```

---

# Part II — Architecture and Codebase Interaction

## 6. End-to-End Flow Overview

```text
config.yaml + params.yaml + schema.yaml
            |
            v
    ConfigurationManager
            |
            v
Pipeline classes (src/datascience/pipeline/)
            |
            v
Component classes (src/datascience/components/)
            |
            v
data/ + artifacts/ + logs/ + MLflow/DagsHub
            |
            v
Flask app (app.py) + PredictionPipeline
```

This separation is important:

- **configuration files** describe what to run
- **entity classes** define the structure of that configuration
- **component classes** do the real work
- **pipeline classes** orchestrate component execution
- **DVC** versions raw and processed datasets under `data/`
- **`main.py`** runs the whole training workflow
- **`app.py`** exposes browser routes for training and prediction

---

## 7. How Each Part Connects to the Others

### 7.1 Configuration Layer

| File | Role | Consumed By |
|------|------|-------------|
| `config/config.yaml` | Paths, URLs, MLflow URI, model name | `ConfigurationManager` |
| `params.yaml` | ElasticNet hyperparameters | `ConfigurationManager` → trainer & evaluator |
| `schema.yaml` | Column names, dtypes, target column | `ConfigurationManager` → validation & trainer |

**`src/datascience/entity/config_entity.py`** defines dataclasses:

- `DataIngestionConfig`
- `DataValidationConfig`
- `DataTransformationConfig`
- `ModelTrainerConfig`
- `ModelEvaluationConfig`

**`src/datascience/config/configuration.py`** reads YAML and builds typed config objects for each stage.

### 7.2 Component Layer (Business Logic)

| Component | Input | Output | Key Dependency |
|-----------|-------|--------|----------------|
| `data_ingestion.py` | `source_URL` from config | `data.zip`, `winequality-red.csv` | `urllib`, `zipfile` |
| `data_validation.py` | extracted CSV + `schema.yaml` columns | `status.txt` | `pandas` |
| `data_transformation.py` | raw CSV | `train.csv`, `test.csv` | `train_test_split` |
| `model_trainer.py` | train/test CSV + params | `model.joblib` | `ElasticNet`, `joblib` |
| `model_evaluation.py` | test CSV + model + params | `metrics.json`, MLflow run, registry | `mlflow`, `.env` credentials |

### 7.3 Pipeline Layer (Orchestration)

Each pipeline follows the same pattern:

```text
ConfigurationManager() → get_*_config() → Component(config) → run method
```

| Pipeline File | Calls |
|---------------|-------|
| `data_ingestion_pipeline.py` | `download_file()`, `extract_zip_file()` |
| `data_validation_pipeline.py` | `validate_all_columns()` |
| `data_transformation_pipeline.py` | `train_test_splitting()` |
| `model_trainer_pipeline.py` | `train()` |
| `model_evaluation_pipeline.py` | `log_into_mlflow()` |
| `prediction_pipeline.py` | `predict()` (inference only) |

Pipelines stay thin so you can test and swap components without rewriting orchestration.

### 7.4 Entry Points

| File | Role |
|------|------|
| `main.py` | Runs all 5 training stages in order |
| `app.py` | Flask server: UI, background training, prediction |
| Individual pipeline `__main__` blocks | Run single stage for debugging |

### 7.5 Artifacts and Logs

| Path | Created By | Used By |
|------|------------|---------|
| `data/raw/` | ingestion | validation, transformation, DVC |
| `data/processed/` | transformation | trainer, evaluator, DVC |
| `artifacts/data_validation/status.txt` | validation | operational check |
| `artifacts/model_trainer/model.joblib` | trainer | evaluator, prediction pipeline |
| `artifacts/model_evaluation/metrics.json` | evaluator | manual review, CI checks |
| `logs/logging.log` | logger in `src/datascience/__init__.py` | all stages |
| `logs/training.log` | background `main.py` from `/train` | monitor web-triggered training |

---

## 8. Complete Flow: Ingestion to Prediction

### Stage 1 — Data Ingestion

```text
GitHub ZIP URL
    → download to data/raw/data.zip
    → extract to data/raw/winequality-red.csv
```

**Code path:** `main.py` → `DataIngestionTrainingPipeline` → `DataIngestion`

### Stage 2 — Data Validation

```text
winequality-red.csv
    → read columns
    → compare against schema.yaml COLUMNS
    → write artifacts/data_validation/status.txt
```

**Code path:** `main.py` → `DataValidationTrainingPipeline` → `DataValiadtion`

### Stage 3 — Data Transformation

```text
winequality-red.csv
    → train_test_split (75% / 25%)
    → data/processed/train.csv
    → data/processed/test.csv
```

**Code path:** `main.py` → `DataTransformationTrainingPipeline` → `DataTransformation`

### Stage 4 — Model Training

```text
train.csv + test.csv + params.yaml
    → separate X and y (target = quality)
    → fit ElasticNet
    → save artifacts/model_trainer/model.joblib
```

**Code path:** `main.py` → `ModelTrainerTrainingPipeline` → `ModelTrainer`

### Stage 5 — Model Evaluation and Tracking

```text
test.csv + model.joblib
    → predict on test features
    → compute rmse, mae, r2
    → save metrics.json
    → log to DagsHub MLflow
    → register ElasticnetModel
    → tag model version with metrics
```

**Code path:** `main.py` → `ModelEvaluationTrainingPipeline` → `ModelEvaluation`

### Stage 6 — Prediction (Serving)

```text
User fills form on index.html (11 features)
    → POST /predict
    → app.py builds numpy array (1, 11)
    → PredictionPipeline loads model.joblib
    → model.predict()
    → results.html shows predicted quality
```

**Code path:** `app.py` → `PredictionPipeline` → saved model

### Optional — Background Training via Web

```text
GET /train
    → subprocess.Popen(["python", "main.py"])
    → stdout/stderr → logs/training.log
    → returns 202 immediately

GET /train/status
    → "running" or "idle"
```

This design prevents Gunicorn worker timeout during long training runs.

---

## 9. Model and Tracking Behavior

This project trains an `ElasticNet` regressor on wine quality data.

During evaluation, the project logs:

- parameters from `params.yaml`
- metrics: `rmse`, `mae`, `r2`
- the trained model artifact
- a registered model in DagsHub-hosted MLflow

The registered model name is: **`ElasticnetModel`**

The evaluation flow also tags the registered model version with metric values so the model registry is more informative.

**Credential flow:**

1. `.env` holds `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
2. `model_evaluation.py` loads `.env` if env vars are not already set
3. `mlflow.set_tracking_uri()` points to DagsHub
4. Run is created, metrics logged, model registered

---

## 10. Repository Structure

```text
datascienceproject/
|-- app.py
|-- main.py
|-- medium.md          ← this guide
|-- README.md          ← operational reference
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

---

# Part III — Operations Guide

> The sections below preserve the operational content from `README.md` so this file remains a single source of truth for day-to-day use.

## 11. How the Sections Relate to Each Other

### 11.1 `config/config.yaml`

Main runtime configuration. Defines:

- where downloaded data goes
- where transformed data goes
- where the trained model is saved
- where evaluation output is saved
- which MLflow tracking URI to use
- which registered model name to use

Current model registration name:

```yaml
registered_model_name: ElasticnetModel
```

### 11.2 `params.yaml`

Stores model hyperparameters:

```yaml
ElasticNet:
  alpha: 0.2
  l1_ratio: 0.1
```

### 11.3 `schema.yaml`

Defines expected dataset columns and target column:

```yaml
TARGET_COLUMN:
  name: quality
```

### 11.4 `src/datascience/entity/config_entity.py`

Dataclasses that turn raw YAML values into typed Python objects.

### 11.5 `src/datascience/config/configuration.py`

Configuration manager. Reads all YAML files and creates stage-specific config objects.

### 11.6 `src/datascience/components/`

Business logic:

- `data_ingestion.py` — download and extract ZIP
- `data_validation.py` — schema column check
- `data_transformation.py` — train/test split
- `model_trainer.py` — train ElasticNet, save joblib
- `model_evaluation.py` — metrics, MLflow, registry

### 11.7 `src/datascience/pipeline/`

Thin orchestration wrappers for each component.

### 11.8 `main.py`

End-to-end training entrypoint. Runs stages 1–5 in order.

### 11.9 `app.py`

Flask application:

| Route | Purpose |
|-------|---------|
| `/` | Home page |
| `/train` | Start background training |
| `/train/status` | `running` or `idle` |
| `/predict` | Form-based prediction |

### 11.10 `artifacts/` and `logs/`

Generated at runtime. See Part II for artifact flow.

---

## 12. Prerequisites

- Python 3.11 or compatible
- Git
- Docker Desktop and Docker Compose (for containerized runs)
- DVC CLI (installed by `pip install -r requirements.txt`)
- DagsHub account
- Access to the DagsHub repository used for MLflow logging

---

## 13. Local Setup

### 13.1 Create a virtual environment

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

### 13.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 13.3 Create the local `.env`

```env
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
```

Do not commit `.env`.

### 13.4 DVC note

This repository now tracks dataset directories with DVC:

- `data/raw.dvc`
- `data/processed.dvc`

Commit these DVC metadata files:

- `data/raw.dvc`
- `data/processed.dvc`
- `data/.gitignore`
- `.dvc/config`
- `.dvc/.gitignore`
- `.dvcignore`

Do not commit:

- `.dvc/config.local`
- `.env`
- actual files inside `data/raw/`
- actual files inside `data/processed/`

If your DVC remote is already configured, run:

```bash
dvc pull
```

If a DVC remote is not configured yet, `python main.py` can still recreate the local data from the configured source URL. That command recreates:

- `data/raw/`
- `data/processed/`

This does not replace a shared remote-backed DVC workflow.

---

## 14. DagsHub and MLflow Setup

### 14.1 Connect Git project to DagsHub

**Option 1:** Connect existing GitHub repo from DagsHub UI.

**Option 2:** Add DagsHub as Git remote:

```bash
git remote add dagshub https://dagshub.com/<username>/<repo-name>.git
git push dagshub main
```

MLflow URI for this project:

`https://dagshub.com/faizulkhan56/datascienceproject.mlflow`

Matching DagsHub Git repository:

`https://dagshub.com/faizulkhan56/datascienceproject.git`

Practical sequence for an existing GitHub project:

1. Create or open the matching repository in DagsHub
2. Use DagsHub's "connect existing repository" flow or add DagsHub as an additional Git remote from your local clone
3. Confirm your code is visible in DagsHub after a normal Git push
4. Configure `.env` locally for MLflow credentials
5. Run `python main.py` and verify experiment runs appear in DagsHub MLflow
6. Configure the DVC remote and push tracked data if you also want dataset versioning visible in DagsHub

### 14.2 MLflow configuration

Tracking URI is set in `config/config.yaml`. Credentials load from `.env`.

### 14.3 Finding `MLFLOW_TRACKING_PASSWORD`

Use a DagsHub access token (recommended over account password):

1. Sign in to DagsHub
2. User settings → tokens
3. Create or copy token
4. Set as `MLFLOW_TRACKING_PASSWORD` in `.env`

Token page: `https://dagshub.com/user/settings/tokens`

### 14.4 Verify DagsHub connection

After successful training, confirm:

- new MLflow experiment run
- registered model `ElasticnetModel`
- new model version
- metrics `rmse`, `mae`, `r2`

### 14.5 DVC and DagsHub data versioning

MLflow experiment tracking and DVC data versioning are separate features.

This project now keeps DVC-tracked datasets in:

- `data/raw/`
- `data/processed/`

To make dataset versions visible in DagsHub's data area, you still need to complete the account-specific remote setup on your side:

1. Configure a DVC remote for your storage backend or DagsHub-supported workflow
2. Authenticate that remote locally
3. Run `dvc push`
4. Commit and push the resulting DVC metadata files

### 14.6 Configure DVC remote with DagsHub Storage

After your Git repository is already connected to DagsHub and MLflow tracking is working, configure the DVC remote in this order.

Browser steps:

1. Open your DagsHub repository homepage
2. Find the repository storage / remote / data-connection area
3. Open the data-storage instructions for `DVC`
4. Copy the repository-specific setup values shown there

Terminal steps:

```bash
./.venv/Scripts/dvc.exe remote add -d dagshub s3://dvc
./.venv/Scripts/dvc.exe remote modify dagshub endpointurl https://dagshub.com/<username>/<repo>.s3
./.venv/Scripts/dvc.exe remote modify --local dagshub access_key_id <token>
./.venv/Scripts/dvc.exe remote modify --local dagshub secret_access_key <token>
```

Notes:

- `dagshub` is the DVC remote name and is intentionally separate from Git `origin`
- `-d` makes it the default DVC remote
- `--local` stores credentials in `.dvc/config.local`, which must not be committed

Verify the remote:

```bash
./.venv/Scripts/dvc.exe remote list
```

Push the tracked dataset:

```bash
./.venv/Scripts/dvc.exe push -r dagshub
```

Or, because it is the default:

```bash
./.venv/Scripts/dvc.exe push
```

Verify after push:

```bash
./.venv/Scripts/dvc.exe status
git status
```

Expected outcome:

- DVC status is up to date
- Git status remains clean or only shows intentional tracked config changes
- `.dvc/config.local` stays untracked
- actual files under `data/raw/` and `data/processed/` stay out of Git

Then refresh the DagsHub repository page and check the data-related view again.

---

## 15. Run Locally Without Docker

### 15.1 Full training pipeline

```bash
python main.py
```

### 15.2 Flask app

```bash
python app.py
```

Open: `http://localhost:8080/`

### 15.3 Browser routes

- `http://localhost:8080/` — home
- `http://localhost:8080/train` — background training
- `http://localhost:8080/train/status` — status check

---

## 16. Run with Docker

### 16.1 Why Docker

Consistent environment: Python, dependencies, Gunicorn, port mapping, volume mounts, healthcheck.

### 16.2 Key files

- **`Dockerfile`** — builds image, runs Gunicorn on port 8080
- **`docker-compose.yml`** — loads `.env`, maps ports, mounts `data/`, `artifacts/`, and `logs/`
- **`.dockerignore`** — excludes venv, data payloads, logs, artifacts, `.env` from build context

### 16.3 Build and start

```bash
docker compose build --no-cache
docker compose up -d
```

### 16.4 Check status and logs

```bash
docker compose ps
docker compose logs -f web
```

### 16.5 Custom port

PowerShell:

```powershell
$env:APP_HOST_PORT="8082"
docker compose up -d
```

### 16.6 Stop

```bash
docker compose down
```

---

## 17. Recommended Sequence for a New User

1. Clone the repository
2. Create `.env`
3. Install dependencies or use Docker
4. If a DVC remote already exists, run `dvc pull`; otherwise run `python main.py` to recreate `data/raw/` and `data/processed/`
5. Configure DVC remote access if you want shared data versioning
6. Review `config/config.yaml`, `params.yaml`, `schema.yaml`
7. Start the application with `docker compose up -d` or `python app.py`
8. Confirm data in `data/` and artifacts in `artifacts/`
9. Confirm MLflow logs on DagsHub
10. Test `/predict` and background training

---

## 18. How to Check Results

### 18.1 Local artifacts

After successful training:

- `data/raw/data.zip`
- `data/raw/winequality-red.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `artifacts/model_trainer/model.joblib`
- `artifacts/model_evaluation/metrics.json`
- `data/raw.dvc`
- `data/processed.dvc`

### 18.2 Logs

```bash
Get-Content .\logs\logging.log -Wait
Get-Content .\logs\training.log -Wait
Get-Content .\logs\training.log -Tail 50
```

### 18.3 DagsHub / MLflow

On the run page verify: `rmse`, `mae`, `r2`, parameters, model artifact.

In model registry verify: `ElasticnetModel`, latest version, tags.

---

## 19. How to Test the Project

### 19.1 Syntax check

```bash
python -m py_compile app.py
python -m py_compile main.py
```

### 19.2 Pipeline check

```bash
python main.py
```

Success: no exception, data directories created or reused, metrics JSON written, DagsHub run created.

### 19.3 Web app check

- `/` loads
- `/train` starts background job
- `/train/status` transitions `running` → `idle`
- `/predict` returns prediction

### 19.4 Docker check

```bash
docker compose build --no-cache
docker compose up -d
docker compose ps
docker compose logs -f web
```

---

## 20. Common Operational Checks

| Check | How |
|-------|-----|
| Training running? | `http://localhost:8080/train/status` |
| Training done? | `logs/training.log` — look for "Model Evaluation stage completed" |
| Model exists? | `dir artifacts\model_trainer` |
| Metrics exist? | `type artifacts\model_evaluation\metrics.json` |
| Raw data exists? | `dir data\raw` |
| Processed data exists? | `dir data\processed` |

---

## 21. Troubleshooting

### `/train` hangs or errors

- `/train` uses `subprocess.Popen` — it should return immediately
- Monitor `logs/training.log`
- Rebuild Docker if using containers

### Gunicorn worker timeout

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### No MLflow logs on DagsHub

Check `.env`, username, token, and `mlflow_uri` in `config/config.yaml`.

### Metrics not obvious on model version page

Metrics live primarily on the **MLflow run** page. Model version links back to the run. This project also tags versions with metric values.

### `uv` warning in MLflow logs

Non-fatal. MLflow falls back if `uv` is unavailable.

### Git warning in Docker

Image installs `git`; `GIT_PYTHON_REFRESH=quiet` reduces noise.

---

## 22. What a Successful Run Looks Like

- all five pipeline stages complete
- raw data under `data/raw/`
- processed data under `data/processed/`
- `metrics.json` saved
- MLflow run URL in logs
- new `ElasticnetModel` version created
- `/train/status` returns `idle`
- `/predict` works with trained model

---

## 23. Recommended Daily Workflow

1. Pull latest code
2. Update config or parameters if needed
3. Start Docker or local environment
4. Trigger training
5. Monitor logs
6. Check DagsHub MLflow results
7. Test prediction UI
8. Commit only intended changes

---

## 24. Useful Commands Reference

```bash
pip install -r requirements.txt
dvc pull
./.venv/Scripts/dvc.exe remote add -d dagshub s3://dvc
./.venv/Scripts/dvc.exe remote modify dagshub endpointurl https://dagshub.com/<username>/<repo>.s3
./.venv/Scripts/dvc.exe remote modify --local dagshub access_key_id <token>
./.venv/Scripts/dvc.exe remote modify --local dagshub secret_access_key <token>
./.venv/Scripts/dvc.exe remote list
./.venv/Scripts/dvc.exe push -r dagshub
python main.py
python app.py
docker compose build --no-cache
docker compose up -d
docker compose logs -f web
docker compose down
Get-Content .\logs\training.log -Wait
```

Training status URL: `http://localhost:8080/train/status`

---

## 25. External References

- DagsHub experiment tracking: https://dagshub.com/docs/feature_guide/experiment_tracking/
- DagsHub track ML experiments: https://dagshub.com/docs/use_cases/track_ml_experiments/
- DagsHub connect existing repo: https://dagshub.com/docs/quick_start/connect_existing_project/
- MLflow docs: https://mlflow.org/docs/latest/
- Wine Quality dataset (UCI): https://archive.ics.uci.edu/ml/datasets/wine+quality
- scikit-learn ElasticNet: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

---

## 26. Quick Reference — One Page Summary

```text
THEORY
  Dataset: 11 chemistry features → predict wine quality (regression)
  Model:   ElasticNet (L1 + L2 regularized linear regression)
  Metrics: RMSE ↓, MAE ↓, R² ↑ = better predictions

FLOW
  ingest → validate → split → train → evaluate → serve

CONFIG
  config.yaml  = paths & MLflow
  params.yaml  = alpha, l1_ratio
  schema.yaml  = columns & target

DATA
  DVC tracks: data/raw and data/processed
  artifacts/: model, metrics, validation status, logs remain runtime outputs

RUN
  dvc pull           # optional, if remote is configured
  python main.py     # full pipeline
  python app.py      # web UI
  docker compose up  # containerized

IMPROVE
  tune params → scale features → engineer features → try other models → log all in MLflow
```

---

## 27. Final Notes

This repository is structured so a new user can:

1. understand the theory behind the dataset, model, and metrics
2. trace how every file and stage connects
3. set up credentials and run the training flow
4. inspect artifacts and DagsHub MLflow results
5. serve predictions through the browser
6. iterate toward better wine quality predictions with measurable metric gains

Keep `.env` local, rebuild Docker after runtime changes, and monitor `logs/training.log` plus DagsHub MLflow for a smooth end-to-end experience.

Minimal rebuild path from zero:

1. Clone the repository
2. Create `.env`
3. Install dependencies with `pip install -r requirements.txt`
4. Run `dvc pull` if the DVC remote is already configured; otherwise run `python main.py`
5. Start with `docker compose up -d` or `python app.py`

For day-to-day commands, `README.md` and this `medium.md` stay aligned — use **this file** when you need theory and architecture, and either file for operations.
