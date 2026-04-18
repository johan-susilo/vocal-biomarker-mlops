import sys
import pandas as pd
import parselmouth
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging

# add the 'src' folder to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from features.acoustics import preprocess_sound, features_sustain, features_glide


# -------------------------
# 1. Setup logging
# -------------------------
LOG_FILE = Path("log/extract.log")
LOG_FILE.parent.mkdir(parents = True, exist_ok = True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # prints to console too
    ]
)

logger = logging.getLogger(__name__)


def parse_filename(filepath: str) -> dict:
  """
  Extracts biological and hardware metadata from the standardized audio filename.
  
  Args:
      filepath (str): Full or partial path to the audio file.
      
  Returns:
      dict: Parsed metadata.
  """
  filepath = str(filepath)
  # ignore folders and .wav extension
  stem = Path(filepath).stem
  
  # remove all extensions like .WAV.WAV
  while "." in stem:
    new_stem = Path(stem).stem
    if new_stem == stem:
      break
    stem = new_stem
  
  parts = stem.rsplit("_", 4)
  
  # check if filename contain 5 information needed
  if len(parts) != 5:
    return None # Better to return None and filter out than crash the whole pool
  
  return {
    "filepath": str(filepath),
    "student_id": parts[0].upper(), # force uppercase
    "condition": parts[1].upper(),
    "device": parts[2].upper(),
    "task": parts[3].lower(),
    "repetition": int(parts[4])
  }  
  
def extract_bio_features(filepath: str) -> dict:
  """
  The 'Heavy' function: Opens audio and extracts PRAAT features.  
  """
  # this prevents C++ binding errors if a Path object is passed in.
  filepath = str(filepath)
  
  try:
    # get metadata
    meta = parse_filename(filepath)
    if not meta:
      return None
    
    snd = parselmouth.Sound(filepath)
    # if stereo than just use right channel
    if snd.get_number_of_channels() > 1:
        snd = snd.extract_channel(1)
    snd = preprocess_sound(snd)
    
    if meta["task"] == "p":
      feats = features_glide(snd)
    else:
      feats = features_sustain(snd)
     
    # merge the math dictionary with the metadata dictionary 
    return meta | feats
    
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

if __name__ == "__main__":
  try:
    # setup paths
    RAW_DATA_DIR = Path("data/raw")
    OUTPUT_FILE = Path("data/processed/features_raw.parquet")

    logger.info("Starting feature extraction pipeline")

    # collect a small sample  to test logic and change to .wav
    all_files = [
      f for f in RAW_DATA_DIR.rglob("*")
      if f.is_file() and f.suffix.lower() == ".wav"
    ]
    

    logger.info(f"Found {len(all_files)} .wav files")

    if len(all_files) == 0:
      logger.warning("No audio files found. Exiting.")
      raise SystemExit()

    # run the pipeline
    df = run_extraction_pipeline(all_files, workers=4)

    # save the data
    save_processed_data(df, str(OUTPUT_FILE))
    logger.info(f"Saved parquet file to {OUTPUT_FILE}")
        
    # Check if file exists
    if OUTPUT_FILE.exists():
      logger.info("Success! Parquet file created.")
      
  except Exception as e:
    logger.exception("Pipeline failed with an unexpected error")