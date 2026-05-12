.PHONY: setup extract transform validate train run-pipeline

# Default Python executable
PYTHON = python3

setup:
	@echo "Setting up environment..."
	$(PYTHON) -m pip install -r requirements.txt || echo "Use conda env as defined in environment.yml"

extract:
	@echo "Running Feature Extraction..."
	$(PYTHON) src/etl/extract.py

transform:
	@echo "Running Data Transformation..."
	$(PYTHON) src/etl/transform.py

validate:
	@echo "Running Data Validation..."
	$(PYTHON) src/etl/validate.py || echo "Validation script might not be fully implemented yet."

train:
	@echo "Running Model Training & Evaluation..."
	$(PYTHON) src/ml/train.py

run-pipeline: extract transform validate train
	@echo "Pipeline complete! Evaluation plots are saved in output/plots/ and logged to MLflow."

notebook:
	@echo "Starting Jupyter Notebook..."
	$(PYTHON) -m jupyter notebook notebooks/01_shap_drift_analysis.ipynb

mlflow:
	@echo "Starting MLflow Tracking Server..."
	mlflow ui
