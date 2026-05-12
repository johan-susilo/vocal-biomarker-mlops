import pandas as pd
from pathlib import Path
import logging

# -------------------------
# 1. Setup logging
# -------------------------
LOG_FILE = Path("log/transform.log")
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

def clean_demographics(csv_path: str) -> pd.DataFrame:
  """
  Loads and cleans the participant metadata table.
  Standardizes casing and string formatting for ML encoding.
  """
  
  df_demo = pd.read_csv(csv_path, encoding='utf-8')
  
  # force upper case and hidden trailing
  df_demo['smartphone_model'] = df_demo['smartphone_model'].astype(str).str.upper().str.strip()
  df_demo['sex'] = df_demo['sex'].astype(str).str.upper().str.strip()
  df_demo['student_id'] = df_demo['student_id'].astype(str).str.upper().str.strip()
  
  # only keep column needed for ML
  cols_to_keep = ['student_id', 'age', 'sex', 'smartphone_model']
  df_demo = df_demo[[col for col in cols_to_keep if col in df_demo.columns]]

  return df_demo

def create_ml_dataset(features_path: str, demo_path: str, output_path: str):
  """
  Merges biology with demographics, then aligns X (Noisy) and y (Clean).
  """
    
  # load the raw features
  logger.info("Loading Parquet features and Demographics CSV...")
  df_features = pd.read_parquet(features_path)
  df_demo = clean_demographics(demo_path)
  
  # join demographics info to biological features
  df_full = pd.merge(df_features, df_demo, on='student_id', how='left')
  
  # split into Target (y) and Input (X)
  df_clean = df_full[(df_full['device'] == 'A') & (df_full['condition'] == 'Q')].copy()
  df_noisy = df_full[(df_full['device'].isin(['B', 'C'])) & (df_full['condition'] == 'N')].copy()
  
  logger.info(f"Target (Clean RODE) rows: {len(df_clean)}")
  logger.info(f"Input (Noisy Phone) rows: {len(df_noisy)}")
  
  # composite keys
  merge_keys = ['student_id', 'task', 'repetition', 'age', 'sex', 'smartphone_model']
  
  # merge them to align biological events
  df_ml = pd.merge(
    left = df_noisy,
    right = df_clean,
    on = merge_keys, 
    suffixes = ('_noisy', '_clean')
  )
  
  # save the final ML-ready dataset
  Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  df_ml.to_parquet(output_path, engine="pyarrow", index=False)
  
  csv_output_path = output_path.replace(".parquet", ".csv")
  df_ml.to_csv(csv_output_path, index=False)
  
  logger.info(f"ML Dataset aligned and saved! Final rows aligned: {len(df_ml)}")
  logger.info(f"CSV file saved to: {csv_output_path}")
  return df_ml

# if run directly run below
if __name__ == "__main__":
  FEATURES_IN = "data/processed/features_raw.parquet"
  DEMO_IN = "data/raw/participants.csv"
  ML_OUT = "data/processed/ml_dataset.parquet"
  
  try:
    create_ml_dataset(FEATURES_IN, DEMO_IN, ML_OUT)
  except Exception as e:
    logger.exception("Transform pipeline failed!")
  