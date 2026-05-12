import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import numpy as np
import optuna
import warnings
import logging
import json
from pathlib import Path

warnings.filterwarnings('ignore')
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# -------------------------
# 1. Setup logging
# -------------------------
LOG_FILE = Path("log/optimize.log")
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

# tell Optuna to also use our logging format instead of its default print statements
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def load_data():
    df = pd.read_parquet("data/processed/ml_dataset.parquet")
    categorical_cols = ['sex', 'smartphone_model_noisy', 'task']
    for col in categorical_cols:
       df[col] = df[col].astype('category')
        
    features = ['f0_mean_noisy', 'jitter_local_noisy', 'shimmer_local_noisy', 
                'F1_mean_noisy', 'F2_mean_noisy', 'sex', 'smartphone_model_noisy', 'task']
    target = 'target_f0_delta'
    
    df['target_f0_delta'] = df['f0_mean_clean'] - df['f0_mean_noisy']

    df_clean = df.dropna(subset=features + [target]).copy()
    return df_clean[features], df_clean[target], df_clean['student_id'], categorical_cols
  
# load data once (outside the loop to save time)
X, y, groups, cat_cols = load_data()

def objective(trial):
  """
  This function is Optuna's playground. It will run this function dozens of times,
  injecting different parameters each time to see what happens.
  """
  
  # define search space
  xgb_params = {
   "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300),
   "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
   "max_depth": trial.suggest_int("xgb_max_depth", 2, 6),
   "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
   "enable_categorical": True,
   "random_state": 42
    }
    
  # CatBoost Search Space
  cat_params = {
    "iterations": trial.suggest_int("cat_iterations", 100, 500),
    "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.2, log=True),
    "depth": trial.suggest_int("cat_depth", 3, 8),
    "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1, 10),
    "cat_features": cat_cols,
    "verbose": 0,
    "random_seed": 42
  }
  
  
  
  gkf = GroupKFold(n_splits=5)
  xgb_model = XGBRegressor(**xgb_params)
  cat_model = CatBoostRegressor(**cat_params)
  rmse_scores = []
  
  
  # evaluate the parameters using our GroupKFold strategy
  for train_idx, test_idx in gkf.split(X, y, groups):
    
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
      
      # new model instances for every fold
      m_xgb = XGBRegressor(**xgb_params)
      m_cat = CatBoostRegressor(**cat_params)
      
      m_xgb.fit(X_train, y_train)
      m_cat.fit(X_train, y_train)
        
      # mathematically average their predictions
      preds = (m_xgb.predict(X_test) + m_cat.predict(X_test)) / 2.0
      rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
      
  # return the Average RMSE. Optuna's goal is to make this number as small as possible.
  return np.mean(rmse_scores)
  
if __name__ == "__main__":
  try:
    logger.info("Starting Optuna Hyperparameter Search...")

    study = optuna.create_study(direction="minimize", study_name="xgboost_debiasing")
    study.optimize(objective, n_trials=50) 
    
    logger.info("Optimization Complete!")
    logger.info(f"Best RMSE Score: {study.best_value:.3f} Hz")
        
    logger.info("Best Parameters Found:")
    for key, value in study.best_params.items():
      logger.info(f"    {key}: {value}")
        
    # save the best parameters to a JSON file so train.py can find them
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
        
    params_file = model_dir / "best_params.json"
    with open(params_file, "w") as f:
        json.dump(study.best_params, f, indent=4)
            
    logger.info(f"Best parameters saved dynamically to {params_file}")

  except Exception as e:
    logger.exception("Optuna optimization pipeline failed!")
  