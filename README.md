# [ARISA-MLOps] 🩺Diabetes Classification Project

Diabetes Classifier based on Kaggle dataset: [Diabetes Database](https://www.kaggle.com/datasets/shahnawaj9/diabetes-database). The project implements a classification pipeline to predict whether a person has diabetes based on a set of medical indicators.

The aim of the project is to create an efficient classification model predicting the occurrence of type 2 diabetes in the population based on medical data. Key performance indicators (KPIs) are:
- F1-score (weighted) ≥ 0.75 (metric recorded in MLflow)
- Logloss as an indicator of the quality of probabilistic classification
- Stability of metrics in cross-validation (std) ≤ 0.05
- Automation of the hyperparameter selection process using Optuna
- Tracking of code and model changes via MLflow + Git

**Risk Assessment:**
- Data bias risk: lack of demographic features may lead to limited generalization
- Model dependency risk: use of specific framework (CatBoost)
- Overfitting – controlled by early stopping and cross-validation
- Reproducibility risk – minimized by version control joblib, mlflow, git hash, random_state


## Dataset Description
The dataset consists of **768 samples** and **9 columns** (8 features + 1 label). The database did not contain missing values ​​and outliers. The data was divided into two sets: training (80%) and testing (20%). Below is a description of each column:
| Column                    | Description                                 |
|---------------------------|---------------------------------------------|
| **Pregnancies**           | Number of times the patient was pregnant    |
| **Glucose**               | Plasma glucose concentration                |
| **BloodPressure**         | Diastolic blood pressure (mm Hg)            |
| **SkinThickness**         | Triceps skin fold thickness (mm)           |
| **Insulin**               | 2-Hour serum insulin (mu U/ml)              |
| **BMI**                   | Body Mass Index (weight in kg/(height in m)^2) |
| **DiabetesPedigreeFunction** | Diabetes pedigree function (heredity factor) |
| **Age**                   | Age of the patient (in years)              |
| **Outcome**               | Diabetes diagnosis result (1 = positive, 0 = negative) |

## Model Description

The classifier uses the **CatBoostClassifier model** (gradient boosting algorithm). The characteristic feature of the model is native support for categorical data without the need for prior encoding (e.g. by one-hot encoding). Instead, CatBoost uses internal mechanisms based on order statistics, which allow to preserve data properties and reduce the risk of overfitting. This model is also distinguished by its resistance to overfitting thanks to the use of the so-called ordered boosting, a special procedure for creating training sets in such a way as to avoid information leakage between examples.

### Hyperparameter tuning:
- Implemented using Optun and mlflow.start_run(nested=True)
- Parameters are saved to the best_params.pkl file and logged in MLflow

### Cross-validation
- Implemented by catboost.cv with 5-fold stratified shuffle
- Results (F1 and logloss) are visualized with standard errors using Plotly

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ARISA_DSML and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ARISA_DSML   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ARISA_DSML a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Setup
### Prerequisites
The project was developed using the following technologies and tools:

- **Python 3.11+** — main programming language
- **Pandas** and **NumPy** — data manipulation and numerical operations
- **Scikit-learn** — machine learning models and evaluation metrics
- **Matplotlib** and **Seaborn** — data visualization
- **Jupyter Notebook** — interactive development environment
- **MLflow** — experiment tracking 
- **Git & GitHub** — version control and collaboration

### How to Run the Project
**1. Clone the repository**
```bash
git clone https://github.com/KK-MUT/MLOps-Final.git
cd MLOps-Final
```
**2. Create and activate virtual environment**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```
**3. Install dependencies**
```bash
pip install -r requirements.txt
```
**4. Kaggle authentication**
To access datasets or interact with Kaggle via the command line or scripts, you need to configure your Kaggle API credentials.
1. Sign in to your Kaggle account at [https://www.kaggle.com](https://www.kaggle.com).
2. Click on your profile picture (top-right corner), then go to **"Account"**.
3. Scroll down to the **API** section and click **"Create New API Token"**.
4. This will automatically download a file named `kaggle.json` — it contains your API credentials.
5. Move the `kaggle.json` file to the appropriate location depending on your OS:

   - **Windows**:  
     `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     
   - **Linux/macOS**:  
     `/home/<your-username>/.config/kaggle/kaggle.json`

**5. Experiment tracking**
Then open http://localhost:5000 in your browser to view logged experiments.
```bash
mlflow ui
```

## Traceability and Reproducibility
### Infrastructure traceability and reproducibility
- Infrastructure defined as code (IaC), stored in a Git repository (.github/workflows)
- Introducing changes only via pull requests, with automatic deployment via GitHub Actions
- Securing the main branch from direct commits
- Pipeline CI/CD as the only source of implementing changes
- Configuring two environments according to the assumptions of the repository and pipelines
- Access of environments to identical data (e.g. consistency of input data in pipelines)

### ML code traceability and reproducibility
- Data processing, model training and API code saved in versioned folder ARISA_DSML/
- All code changes introduced by PRs
- Project environment (venv + pyproject.toml, setup.cfg files) fully reproducible – defined as code
- Unified runtime environment (works locally and on GitHub Actions without modifications)
- Possibility to clearly link model launch to:
   - source code (commit Git),
   - infrastructure (pipeline, environment),
   - training data (CSV file or artifact in MLflow),

## Code Quality
### Infrastructure code quality
- The implemented CI pipeline (GitHub Actions) automatically validates the configuration and runs tests after a change in the code.
- Review by other team members is required before merging changes.
### ML model code quality
- pre-commit hooks (flake8, ruff) have been implemented in the repository.
- The ML code has been divided into modules and includes unit tests for:
    - data processing,
    - model training,
    - predictions.
- After each PR, the CI pipeline containing linting (lint-code.yml) is run.
- When data (test.csv, train.csv), prediction code (predict.py) or model (catboost_model_diabetes.cbm) is changed, the CI pipeline containing the prediction (predict_on_model_change.yml) is run.
- When data (test.csv, train.csv), data preparation code (preproc.py) or training code (train.py) or best model parameters (best_params.pkl) are changed, the CI pipeline responsible for data preparation and training (retrain_on_change.yml) is run.
- he code documentation is stored in the repository as documentation-as-code – in Markdown files and function docstrings.
- Release notes (description in PR) are created for each version of the model code.

## Monitoring & Support
### Infrastructure Monitoring Requirements
An alert system (e.g. pipeline errors, failed implementations, execution timeouts) allows you to quickly identify problems and respond.
### Application Monitoring Requirements
Each pipeline execution is automatically recorded (duration, success/failure, console logs).
### KPI & Model Performance Monitoring Requirements
- Validation metrics (e.g. F1-score, Accuracy, LogLoss) from historical data are recorded and monitored every time you run a model training.
- Results from MLflow allow you to track the history of model performance.
### Data Drift & Outliers Monitoring
- Distributions of the main model features (e.g. age, glucose level, BMI in the diabetes classifier) ​​can be regularly updated and compared to training data.
- In the case of data drift, an alert or a model retraining trigger is triggered.
