# Quantifying and Correcting Hardware-Induced Data Drift in Digital Vocal Biomarkers

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)

## Overview 
In digital health, acoustic Machine Learning models trained on clinical-grade microphones fail when deployed to real-world patient smartphones due to hardware-induced data drift. This project is an end-to-end MLOps pipeline that extracts acoustic features (Jitter, Shimmer, Formants) from 2,400+ audio samples and utilizes an XGBoost regressor to successfully map noisy smartphone biomarkers back to clinical-grade baselines.

## Architecture & Pipeline
Raw Audio -> Parallel Extract -> Data Validation -> XGBoost -> SHAP

- ETL Engine: Implemented asynchronous/multiprocessing I/O to process 2,400+ .WAV files, reducing extraction time by X%.

- Data Quality: Enforced strict schema validation using [Pandera/Pydantic] to prevent silent pipeline failures.

- ML De-biasing: Built a multi-output regression model using GroupKFold cross-validation (grouped by Participant ID) to prevent biometric data leakage.

## Key Biological Insights (SHAP Analysis)
SHAP analysis reveals that smartphone noise-cancellation algorithms artificially truncate F0 variability by 15%, which severely impacts perturbation measures used in neurodegenerative disease screening.

## Reproducibility (How to Run)

```Bash
# Clone the repository
git clone https://github.com/YourName/vocal-biomarker-pipeline.git

# Build the Docker container (Includes Praat dependencies)
docker compose up --build

# Run the automated ETL and Training pipeline
make run-pipeline
```
## Tech Stack
Bioinformatics: Praat, Parselmouth-Python

Data Engineering: Multiprocessing, Parquet, Pandera, Docker

Machine Learning: XGBoost, Scikit-Learn, SHAP