import pandas as pd
import xgboost as xgb
import numpy as np
import warnings
import logging
import mlflow
import mlflow.xgboost
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# suppress some pandas/xgboost warnings for cleaner output
warnings.filterwarnings('ignore')

# -------------------------
# 1. Setup logging
# -------------------------
LOG_FILE = Path("log/train.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_debiasing_model(parquet_path: str):
  logger.info("Initializing ML Training Pipeline...")
  
  df = pd.read_parquet(parquet_path)
  
  # convert text columns to Pandas "category" type
  categorical_cols = ['sex', 'smartphone_model', 'task']
  for col in categorical_cols:
    df[col] = df[col].astype('category')
  
  # define features (X) and target (y)
  features = [
        'f0_mean_noisy', 
        'jitter_local_noisy', 
        'shimmer_local_noisy', 
        'F1_mean_noisy',
        'F2_mean_noisy',
        'sex', 
        'smartphone_model', 
        'task'
    ]
  target = ['f0_mean_clean']
  
  # drop rows where our target or features are NaN
  df_clean = df.dropna(subset=features + target).copy()
  
  X = df_clean[features]
  y = df_clean[target]
  groups = df_clean['student_id'] # to prevent biometric data leakage
  
  # mlops setup experiment
  mlflow.set_experiment("Smartphone_Hardware_Debiasing")
  
  with mlflow.start_run(run_name="Feature_Engineering_v1"):
    
    # log hyperparams
    params = {
      "n_estimators": 100,
      "learning_rate": 0.05,        # lower learning rate (learns slower and safer)
      "max_depth": 3,               # lower depth (prevents the tree from memorizing specific voices)
      "subsample": 0.8,             # randomize 80% of rows per tree (prevents overfitting to one student)
      "colsample_bytree": 0.8,      # randomize 80% of features per tree
      "enable_categorical": True,
      "random_state": 42
    }
    
    mlflow.log_params(params)
    # also log the features we used so we don't forget!
    mlflow.log_param("features", ", ".join(features))
  
    # setup GroupKFold Cross Validation (5 folds)
    # data points belonging to the same group must stay together
    # important when the data contains groups of instances that are not independent of each other
    gkf = GroupKFold(n_splits=5)
  
    # init model
    model = xgb.XGBRegressor(**params)
  
    r2_scores, rmse_scores = [], []
    logger.info(f"Training on {len(X)} biological events across {df_clean['student_id'].nunique()} students...")
  
    # training loop
    # idx used to track indices of the samples in the dataset
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

      # train
      model.fit(X_train, y_train)

      # predict
      preds = model.predict(X_test)

      # evaluate
      r2 = r2_score(y_test, preds)
      rmse = np.sqrt(mean_squared_error(y_test, preds))

      r2_scores.append(r2)
      rmse_scores.append(rmse)

      mlflow.log_metric(f"fold_{fold+1}_r2", r2)
      mlflow.log_metric(f"fold_{fold+1}_rmse", rmse)
      
      logger.info(f"Fold {fold+1} | R2: {r2:.3f} | RMSE: {rmse:.3f} Hz")
    
  avg_r2 = np.mean(r2_scores)
  avg_rmse = np.mean(rmse_scores)
  
  logger.info("✅ Cross-Validation Complete.")
  logger.info(f"Average R2: {avg_r2:.3f}")
  logger.info(f"Average RMSE: {avg_rmse:.3f} Hz")
  
  mlflow.log_metric("avg_r2", avg_r2)
  mlflow.log_metric("avg_rmse", avg_rmse)
  
  mlflow.xgboost.log_model(model, "xgboost_model")
  logger.info("Run logged successfully to MLflow!")
  
  model_dir = Path("models")
  model_dir.mkdir(parents=True, exist_ok=True)
  
  model_path = model_dir / "xgboost_debias.json"
  model.save_model(model_path)
  
  logger.info(f"Model natively saved to {model_path}")
  
  # generate evaluation plots and log to MLflow
  logger.info("Generating evaluation plots for MLflow...")
  from evaluate import generate_evaluation_plots

  # use the last X_test, y_test from the K-fold for a quick plot,
  # or evaluate on the whole training set for presentation purposes
  scatter, residuals, improvement = generate_evaluation_plots(model, X, y, plot_dir="output/plots")

  mlflow.log_artifact(scatter, "evaluation_plots")
  mlflow.log_artifact(residuals, "evaluation_plots")
  if improvement:
    mlflow.log_artifact(improvement, "evaluation_plots")

  logger.info("Evaluation plots saved and logged as artifacts.")

  return model

if __name__ == "__main__":
  try:
    trained_model = train_debiasing_model("data/processed/ml_dataset.parquet")
  except Exception as e:
    logger.exception("Training pipeline failed!")
  
  
  

  