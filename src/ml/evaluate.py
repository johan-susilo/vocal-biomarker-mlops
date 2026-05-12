import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from pathlib import Path
import logging

# -------------------------
# Setup logging
# -------------------------
LOG_FILE = Path("log/evaluate.log")
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

def generate_evaluation_plots(model, X, y_true, plot_dir="output/plots"):
    """
    Generates and saves evaluation plots for the model.
    """
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X)

    sns.set_theme(style="whitegrid")

    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="royalblue")

    # plot ideal y=x line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal (y=x)")

    plt.xlabel("Clean Baseline F0 (Hz)")
    plt.ylabel("Predicted Clean F0 (Hz)")
    plt.title("Model Prediction vs. Clinical Baseline")
    plt.legend()

    scatter_path = f"{plot_dir}/actual_vs_predicted.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Residuals Distribution
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=50, kde=True, color="seagreen")
    plt.axvline(0, color='r', linestyle='--', lw=2)
    plt.xlabel("Error (Hz)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution (Residuals)")

    residuals_path = f"{plot_dir}/residuals_distribution.png"
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Raw Noisy vs Clean vs Predicted
    if 'f0_mean_noisy' in X.columns:
        plt.figure(figsize=(10, 6))

        # calculate absolute errors
        raw_error = np.abs(X['f0_mean_noisy'] - y_true)
        pred_error = np.abs(y_pred - y_true)

        error_df = pd.DataFrame({
            "Error Type": ["Raw Noisy vs Clean"] * len(raw_error) + ["Predicted vs Clean"] * len(pred_error),
            "Absolute Error (Hz)": np.concatenate([raw_error, pred_error])
        })

        sns.boxplot(x="Error Type", y="Absolute Error (Hz)", data=error_df, palette="Set2")
        plt.title("Improvement: Raw Noisy Phone vs. Corrected Prediction")

        improvement_path = f"{plot_dir}/improvement_boxplot.png"
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        improvement_path = None

    logger.info(f"Evaluation plots saved to {plot_dir}")

    return scatter_path, residuals_path, improvement_path

def evaluate_model(model_path: str, data_path: str, plot_dir="output/plots"):
    """
    Loads model and data, evaluates it, and generates plots.
    """
    logger.info("Loading model and data for evaluation...")

    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    try:
        df = pd.read_parquet(data_path)

        categorical_cols = ['sex', 'smartphone_model', 'task']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        features = [
            'f0_mean_noisy', 'jitter_local_noisy', 'shimmer_local_noisy',
            'F1_mean_noisy', 'F2_mean_noisy', 'sex', 'smartphone_model', 'task'
        ]
        target = ['f0_mean_clean']

        df_clean = df.dropna(subset=features + target).copy()
        X = df_clean[features]
        y = df_clean[target].values.ravel()

    except Exception as e:
        logger.error(f"Failed to load or process data: {e}")
        return

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    logger.info(f"Evaluation R2: {r2:.3f}")
    logger.info(f"Evaluation RMSE: {rmse:.3f} Hz")

    generate_evaluation_plots(model, X, y, plot_dir)

if __name__ == "__main__":
    evaluate_model("models/xgboost_debias.json", "data/processed/ml_dataset.parquet")
