"""
data_cleaning cleans the raw census.csv export, which has a leading space after every
comma (in the header row and in every string-valued field).
"""
import os
import pandas as pd
from utils import logger

 
RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "census.csv"
)
CLEAN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "census_clean.csv"
)
 
 
def clean_census_data(raw_path: str, clean_path: str) -> pd.DataFrame:
    """
    Load the raw census CSV, strip whitespace from column names and
    string cell values, and write the cleaned result to disk.
 
    Parameters
    ----------
    raw_path : str
        Path to the raw, messy census.csv.
    clean_path : str
        Path to write the cleaned CSV to.
 
    Returns
    -------
    pd.DataFrame
        The cleaned dataframe (also written to clean_path).
    """
    try:
        logger.info("Reading csv file")
        df = pd.read_csv(raw_path, skipinitialspace=True)
    
        logger.info("cleaning data...")
        df.columns = [c.strip() for c in df.columns]
    
        str_like_cols = df.select_dtypes(include=["object"]).columns
        for col in str_like_cols:
            df[col] = df[col].astype(str).str.strip()
        
        logger.info("Saving clean data...")
        df.to_csv(clean_path, index=False)
        logger.info(f"Written to: {CLEAN_PATH}")
        return df
    except ValueError as err:
        logger.error(f"data cleaning: error {err}")
    except Exception as e:
        logger.error(f"data cleaning: error {e}")
 
 
if __name__ == "__main__":
    clean_census_data(RAW_PATH, CLEAN_PATH)

 