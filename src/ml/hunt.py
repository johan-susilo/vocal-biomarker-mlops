import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_predict, GroupKFold
import numpy as np

def hunt_outlier():
    print("Hunting for the Fold 1 Outlier...")
    
    # 1. Load Data
    df = pd.read_parquet("data/processed/ml_dataset.parquet")
    
    categorical_cols = ['sex', 'smartphone_model', 'task']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        
    features = ['f0_mean_noisy', 'jitter_local_noisy', 'shimmer_local_noisy', 
                'F1_mean_noisy', 'F2_mean_noisy', 'sex', 'smartphone_model', 'task']
    target = 'f0_mean_clean'
    
    df_clean = df.dropna(subset=features + [target]).copy()
    X = df_clean[features]
    y = df_clean[target]
    groups = df_clean['student_id']

    # 2. Initialize Model (Using basic params for the hunt)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, enable_categorical=True, random_state=42
    )

    # 3. Generate predictions for every row using out-of-fold testing
    gkf = GroupKFold(n_splits=5)
    preds = cross_val_predict(model, X, y, groups=groups, cv=gkf)
    
    # 4. Calculate how wrong the model was for each row
    df_clean['predicted_pitch'] = preds
    df_clean['absolute_error_hz'] = np.abs(df_clean[target] - preds)

    # 5. Group by student to find the culprit
    student_errors = df_clean.groupby(['student_id', 'sex']).agg(
        avg_error_hz=('absolute_error_hz', 'mean'),
        actual_pitch_clean=('f0_mean_clean', 'mean'),
        actual_pitch_noisy=('f0_mean_noisy', 'mean'),
        recordings=('student_id', 'count')
    ).reset_index()

    # Sort to find the highest errors
    student_errors = student_errors.sort_values(by='avg_error_hz', ascending=False)
    
    print("\nTOP 5 HARDEST STUDENTS TO PREDICT (The Outliers):")
    print(student_errors.head(5).to_string(index=False))

if __name__ == "__main__":
    hunt_outlier()