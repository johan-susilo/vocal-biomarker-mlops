# Quantifying and Correcting Hardware-Induced Data Drift in Digital Vocal Biomarkers

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)

## Overview 
In digital health, acoustic Machine Learning models trained on clinical-grade microphones fail when deployed to real-world patient smartphones due to hardware-induced data drift. This project is an end-to-end MLOps pipeline that extracts acoustic features (Jitter, Shimmer, Formants) from 2,400+ audio samples and utilizes an XGBoost regressor to successfully map noisy smartphone biomarkers back to clinical-grade baselines.

## Architecture & Pipeline
Raw Audio -> Parallel Extract -> Data Validation -> XGBoost -> SHAP

The codebase is organized into distinct functional modules:

- **ETL Engine (`src/etl/`)**:
  - `extract.py`: Implements asynchronous/multiprocessing I/O to parse clinical and noisy .WAV files and extract Praat acoustic features.
  - `transform.py`: Cleans demographic metadata and aligns the noisy (input) vs clean (target) biological events into a machine learning dataset.
  - `validate.py`: Enforces strict schema validation using Pandera/Pydantic to prevent silent pipeline failures.

- **ML De-biasing (`src/ml/`)**:
  - `train.py`: Builds a multi-output regression model using GroupKFold cross-validation (grouped by Participant ID) to prevent biometric data leakage. Logs hyper-parameters and models to **MLflow**.
  - `evaluate.py`: Automatically generates and logs visual presentations of model performance, including Actual vs Predicted scatter plots, Residual distributions, and direct improvement comparisons.

- **Presentation & Insights (`notebooks/`)**:
  - `01_shap_drift_analysis.ipynb`: A presentation-ready Jupyter Notebook that computes SHAP values to explain feature importance and visually demonstrates the hardware-induced drift.

## Key Biological Insights (SHAP Analysis)
SHAP analysis reveals that smartphone noise-cancellation algorithms artificially truncate F0 variability by 15%, which severely impacts perturbation measures used in neurodegenerative disease screening.

## Data Setup
To maintain a lightweight repository, raw audio files are symlinked from the HPC SCRATCH storage to data/raw/ using ln -s. The pipeline expects 2,439 .WAV files following the SXX_C_D_t_r naming convention.

## Reproducibility (How to Run)

Ensure you have your environment set up (via Conda `environment.yml` or Docker). You must have the raw `.WAV` data in `data/raw/` and `participants.csv` available.

```Bash
# 1. Clone the repository
git clone https://github.com/YourName/vocal-biomarker-pipeline.git

# 2. Run the automated ETL, Training, and Evaluation pipeline
# This will extract features, align data, train XGBoost, and generate evaluation plots.
make run-pipeline

# 3. View the evaluation plots and ML tracking
# Open http://localhost:5000 in your browser to view the MLflow dashboard
make mlflow

# 4. Explore the SHAP Drift Analysis Presentation
# Opens the interactive Jupyter notebook to view hardware truncation analysis
make notebook
```
Evaluation artifacts (scatter plots, residual distributions) will also be directly available in the `output/plots/` directory after running the pipeline.

## Tech Stack
Bioinformatics: Praat, Parselmouth-Python

Data Engineering: Multiprocessing, Parquet, Pandera, Docker

Machine Learning: XGBoost, Scikit-Learn, SHAP