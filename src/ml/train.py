import pandas as pd
import xgboost as xgb
import numpy as np
import warnings
import logging
import mlflow
import mlflow.xgboost
import json
import matplotlib.pyplot as plt
import joblib
import shap

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
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
  categorical_cols = ['sex', 'smartphone_model_noisy', 'task']
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
        'smartphone_model_noisy', 
        'task'
    ]
  target = ['target_f0_delta']
  
  # drop rows where our target or features are NaN
  df_clean = df.dropna(subset=features + target).copy()
  
  X = df_clean[features]
  y = df_clean[target].values.ravel() # ridge requires 1D
  groups = df_clean['student_id'] # to prevent biometric data leakage
  
  # mlops setup experiment
  mlflow.set_experiment("Smartphone_Hardware_Debiasing")
  
  with mlflow.start_run(run_name="Ensemble_model"):
   
    params_file = Path("models/best_params.json")
    with open(params_file, "r") as f:
        all_params = json.load(f)
        
    xgb_params = {k.replace("xgb_", ""): v for k, v in all_params.items() if k.startswith("xgb_")}
    xgb_params["enable_categorical"] = True
    xgb_params["random_state"] = 42
    
    cat_params = {k.replace("cat_", ""): v for k, v in all_params.items() if k.startswith("cat_")}
    cat_params["cat_features"] = categorical_cols
    cat_params["verbose"] = 0
    cat_params["random_seed"] = 42
    gkf = GroupKFold(n_splits=5)
    
    
    # we will save models from the last fold for production use
    final_xgb = None
    final_cat = None
    
    # training loop
    
    #out of fold predictions
    oof_preds_xgb = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))
    
    # idx used to track indices of the samples in the dataset
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]

      # define the model
      m_xgb = XGBRegressor(**xgb_params)
      m_cat = CatBoostRegressor(**cat_params)
      
      # train
      m_xgb.fit(X_train, y_train)
      m_cat.fit(X_train, y_train)

      # predict
      oof_preds_xgb[test_idx] = m_xgb.predict(X_test)
      oof_preds_cat[test_idx] = m_cat.predict(X_test)
      
    meta_X = np.column_stack((oof_preds_xgb, oof_preds_cat))
    meta_model = Ridge()
    
    meta_model.fit(meta_X, y)
    final_delta_preds = meta_model.predict(meta_X)
    
    predicted_absolute_f0 = X['f0_mean_noisy'].values + final_delta_preds.ravel()
    true_absolute_f0 = df_clean['f0_mean_clean']
    
    final_rmse = np.sqrt(mean_squared_error(true_absolute_f0, predicted_absolute_f0))
    final_r2 = r2_score(true_absolute_f0, predicted_absolute_f0)
    
    # create a dataframe of your predictions vs reality
    eval_df = pd.DataFrame({
        'true_f0': true_absolute_f0,
        'pred_f0': predicted_absolute_f0,
        'device': X['smartphone_model_noisy'].values,
        'sex': X['sex'].values
    })

    eval_df['sq_error'] = (eval_df['true_f0'] - eval_df['pred_f0'])**2

    # log RMSE for Males vs Females to prove it's not hopelessly biased
    for sex_val in eval_df['sex'].unique():
        subset = eval_df[eval_df['sex'] == sex_val]
        rmse_subset = np.sqrt(subset['sq_error'].mean())
        mlflow.log_metric(f"rmse_sex_{sex_val.lower()}", rmse_subset)
    
    logger.info("✅ Cross-Validation Complete.")
    logger.info(f"Final R2: {final_r2:.3f}")
    logger.info(f"Final RMSE: {final_rmse:.3f} Hz")
    
    mlflow.log_metric("final_r2", final_r2)
    mlflow.log_metric("final_rmse", final_rmse)
    
    logger.info("Retraining base models on 100% of the data for production...")
    final_xgb = XGBRegressor(**xgb_params)
    final_cat = CatBoostRegressor(**cat_params)
    
    final_xgb.fit(X, y)
    final_cat.fit(X, y)
    
    
    mlflow.xgboost.log_model(final_xgb, "xgboost_model")
    mlflow.catboost.log_model(final_cat, "catboost_model")
    mlflow.sklearn.log_model(meta_model, "meta_model")
    
    logger.info("Run logged successfully to MLflow!")
    
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    final_xgb.save_model(str(model_dir / "final_xgb.json"))
    final_cat.save_model(str(model_dir / "final_cat.cbm"))
    joblib.dump(meta_model, str(model_dir / "meta_ridge.pkl"))
    
    
    # extract the mathematical weights XGBoost assigned to each column
    importances = final_xgb.feature_importances_
    feature_names = X.columns
    
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10,6))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='#00a699')
    plt.xlabel('Relative Importance (Weight)')
    plt.title('How the AI Corrects Smartphone Audio (Feature Importance)')
    plt.tight_layout()
    
    plot_path = Path("models/feature_importance.png")
    plt.savefig(plot_path)
    mlflow.log_artifact(str(plot_path))
    logger.info(f"Feature Importance chart saved to {plot_path} and logged to MLflow!")
    
    # ---------------------------------------------------
    # DIAGNOSTIC 1: RESIDUAL PLOT (PREDICTED VS TRUE)
    # ---------------------------------------------------
    plt.figure(figsize=(8, 8))

    # Plot the actual predictions
    plt.scatter(true_absolute_f0, predicted_absolute_f0, alpha=0.5, color='#00a699', label='Predictions')

    # Plot the 1:1 perfect prediction line (diagonal)
    min_val = min(true_absolute_f0.min(), predicted_absolute_f0.min())
    max_val = max(true_absolute_f0.max(), predicted_absolute_f0.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Accuracy')

    plt.xlabel('True F0 (Hz)')
    plt.ylabel('Predicted F0 (Hz)')
    plt.title('Residual Analysis: Predicted vs True Pitch')
    plt.legend()
    plt.tight_layout()

    residual_path = Path("models/residuals.png")
    plt.savefig(residual_path)
    mlflow.log_artifact(str(residual_path))
    plt.close() # clear memory
    logger.info(f"Residual plot saved to {residual_path} and logged to MLflow!")
    
    # ---------------------------------------------------
    # DIAGNOSTIC 2: SHAP VALUES (DIRECTIONAL IMPACT)
    # ---------------------------------------------------
    logger.info("Generating SHAP values for model explainability...")

    # TreeExplainer cracks open the XGBoost trees to see exact marginal contributions
    explainer = shap.TreeExplainer(final_xgb)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    # show=False prevents it from hanging the terminal, allowing us to save it
    shap.summary_plot(shap_values, X, show=False) 

    plt.title('SHAP Summary: Directional Impact on Hardware Distortion')
    plt.tight_layout()

    shap_path = Path("models/shap_summary.png")
    plt.savefig(shap_path, bbox_inches='tight')
    mlflow.log_artifact(str(shap_path))
    plt.close()
    logger.info(f"SHAP summary plot saved to {shap_path} and logged to MLflow!")
    
  return final_xgb, final_cat, meta_model

if __name__ == "__main__":
  try:
    trained_model = train_debiasing_model("data/processed/ml_dataset.parquet")
  except Exception as e:
    logger.exception("Training pipeline failed!")
  
  
  

  