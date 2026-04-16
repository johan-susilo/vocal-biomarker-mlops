import pandas as pd
import parselmouth
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def parse_filename(filepath: str) -> dict:
  """
  Extracts biological and hardware metadata from the standardized audio filename.
  
  Args:
      filepath (str): Full or partial path to the audio file.
      
  Returns:
      dict: Parsed metadata.
  """
  
  # ignore folders and .wav extension
  stem = Path(filepath).stem
    
  parts = stem.split("_")
  
  # check if filename contain 5 information needed
  if len(parts) != 5:
    return None # Better to return None and filter out than crash the whole pool
  
  return {
    "filepath": str(filepath),
    "student_id": parts[0],
    "condition": parts[1],
    "device": parts[2],
    "task": parts[3],
    "repetition": int(parts[4])
  }  
  
def extract_bio_features(filepath: str) -> dict:
  """
  The 'Heavy' function: Opens audio and extracts PRAAT features.  
  """
  
  try:
    # load audio into parselmouth
    snd = parselmouth.Sound(filepath)
    
    pitch = snd.to_pitch()
    mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    
    # get metadata
    meta = parse_filename(filepath)
    
    # write new mean pitch info into the metadata list
    if meta:
      meta["mean_pitch"] = mean_pitch
      return meta
    
  except Exception as e:
      print(f"Error processing {filepath}: {e}")
      return None
    
def run_extraction_pipeline(file_paths: list[str], workers: int = 4):
  """
  Orchestrates the parallel extraction using all CPU cores.
  """
  
  # using ProcessPoolExecutor to bypass the GIL
  with ProcessPoolExecutor(max_workers=workers) as executor:
    # map() handles the distribution of file_paths across the workers
    results = list(executor.map(extract_bio_features, file_paths))
    
  clean_results = [r for r in results if r is not None]
  
  return pd.DataFrame(clean_results)

def save_processed_data(df: pd.DataFrame, output_path: str):
  """
  Saves the extracted features to a version-controlled Parquet file.
  """
  
  # create the directory if it doesn't exist
  Path(output_path).parent.mkdir(parents=True, exist_ok = True)
  
  # save using the pyarrow engine for maximum efficiency
  df.to_parquet(output_path, engine="pyarrow", index=False)
  
  print(f"Pipeline complete: {len(df)} records saved to {output_path}")

