import pandas as pd
from pathlib import Path

def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(path)
    return df

def save_data(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)
