from pathlib import Path

def parse_filename(filepath: str) -> dict:
  """
    Extracts biological and hardware metadata from the standardized audio filename.
    
    Args:
        filepath (str): Full or partial path to the audio file.
        
    Returns:
        dict: Parsed metadata.
    """
    
    #ignore folders and .wav extension
    stem = Path(filepath).stem
    
    parts = stem.split("_")
    
    if len(parts) != 5:
      raise ValueError(f"Corrupted filename detected: '{stem}'. Expected 5 parts")
    
    return {
      "student_id": parts[0],
      "condition": parts[1],
      "device": parts[2],
      "task": parts[3],
      "repetition": int(parts[4])
    }  