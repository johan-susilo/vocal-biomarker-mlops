import pandas as pd
import xgboost as xgb
import seaborn as sns
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
    
    # out of fold predictions
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
    
    # ---------------------------------------------------
    # REVISED EVAL_DF (Includes original noisy baseline)
    # ---------------------------------------------------
    eval_df = pd.DataFrame({
        'true_f0': true_absolute_f0,
        'noisy_f0': X['f0_mean_noisy'].values,
        'pred_f0': predicted_absolute_f0,
        'device': X['smartphone_model_noisy'].values,
        'sex': X['sex'].values
    })

    # Calculate actual and absolute errors
    eval_df['uncorrected_error'] = eval_df['true_f0'] - eval_df['noisy_f0']
    eval_df['corrected_error'] = eval_df['true_f0'] - eval_df['pred_f0']
    
    eval_df['abs_uncorrected_error'] = eval_df['uncorrected_error'].abs()
    eval_df['abs_corrected_error'] = eval_df['corrected_error'].abs()

    # log RMSE for Males vs Females to prove it's not hopelessly biased
    eval_df['sq_error'] = eval_df['corrected_error']**2
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
    
    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------
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
    plt.close()
    logger.info(f"Feature Importance chart saved to {plot_path} and logged to MLflow!")
    
    # ---------------------------------------------------
    # RESIDUAL PLOT
    # ---------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(true_absolute_f0, predicted_absolute_f0, alpha=0.5, color='#00a699', label='Predictions')
    
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
    plt.close()
    logger.info(f"Residual plot saved to {residual_path} and logged to MLflow!")

    # ---------------------------------------------------
    # Error Distribution by Smartphone Model
    # ---------------------------------------------------
    logger.info("Generating Plot: Error by Device Boxplot...")
    plt.figure(figsize=(14, 7))
    
    melted_df = eval_df.melt(
        id_vars=['device'],
        value_vars=['abs_uncorrected_error', 'abs_corrected_error'],
        var_name='Error Type',
        value_name='Absolute Error (Hz)'
    )
    melted_df['Error Type'] = melted_df['Error Type'].map({
        'abs_uncorrected_error': 'Uncorrected (Raw Smartphone)',
        'abs_corrected_error': 'Corrected (ML Pipeline)'
    })

    sns.boxplot(
        data=melted_df, 
        x='device', 
        y='Absolute Error (Hz)', 
        hue='Error Type', 
        palette=['#ff9999', '#00a699']
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Hardware Bias Normalization: Absolute Pitch Error by Device')
    plt.tight_layout()
    
    box_path = Path("models/error_by_device_boxplot.png")
    plt.savefig(box_path)
    mlflow.log_artifact(str(box_path))
    plt.close()

    # ---------------------------------------------------
    # Before vs. After Correction (Density Plot)
    # ---------------------------------------------------
    logger.info("Generating Plot: Before/After Density...")
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(eval_df['uncorrected_error'], fill=True, color='#ff9999', label='Original Hardware Error', alpha=0.5)
    sns.kdeplot(eval_df['corrected_error'], fill=True, color='#00a699', label='ML Corrected Error', alpha=0.6)
    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    
    plt.xlabel('Error (True Clinical F0 - Observed/Predicted F0) in Hz')
    plt.ylabel('Density (Concentration of Recordings)')
    plt.title('Data Drift Distribution: Hardware Noise vs. ML Correction')
    plt.legend()
    plt.tight_layout()
    
    density_path = Path("models/before_after_density.png")
    plt.savefig(density_path)
    mlflow.log_artifact(str(density_path))
    plt.close()

    # ---------------------------------------------------
    # Performance Fairness by Sex (Bar Chart)
    # ---------------------------------------------------
    logger.info("Generating Plot: Fairness by Sex Barplot...")
    plt.figure(figsize=(8, 6))

    fairness_data = []
    for sex_val in eval_df['sex'].unique():
        subset = eval_df[eval_df['sex'] == sex_val]
        rmse_uncorr = np.sqrt((subset['uncorrected_error']**2).mean())
        rmse_corr = np.sqrt((subset['corrected_error']**2).mean())
        
        fairness_data.append({'Sex': sex_val, 'Phase': 'Uncorrected Baseline', 'RMSE (Hz)': rmse_uncorr})
        fairness_data.append({'Sex': sex_val, 'Phase': 'ML Corrected', 'RMSE (Hz)': rmse_corr})

    fairness_df = pd.DataFrame(fairness_data)

    sns.barplot(
        data=fairness_df, 
        x='Sex', 
        y='RMSE (Hz)', 
        hue='Phase', 
        palette=['#ff9999', '#00a699']
    )
    plt.title('Algorithm Fairness: Model Accuracy by Biological Sex')
    plt.tight_layout()
    
    fairness_path = Path("models/fairness_by_sex_barplot.png")
    plt.savefig(fairness_path)
    mlflow.log_artifact(str(fairness_path))
    plt.close()

    # ---------------------------------------------------
    # SHAP VALUES
    # ---------------------------------------------------
    logger.info("Generating SHAP values for model explainability...")
    explainer = shap.TreeExplainer(final_xgb)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
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