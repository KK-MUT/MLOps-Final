# [ARISA-MLOps] 🩺Diabetes Classification Project

Diabetes Classifier based on Kaggle dataset: [Diabetes Database](https://www.kaggle.com/datasets/shahnawaj9/diabetes-database). The project implements a classification pipeline to predict whether a person has diabetes based on a set of medical indicators.

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

## Dataset Description
The dataset consists of **768 samples** and **9 columns** (8 features + 1 label). Below is a description of each column:
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
