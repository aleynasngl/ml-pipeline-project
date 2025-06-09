import pandas as pd
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def get_data_info(data: pd.DataFrame) -> None:
    """
    Print basic information about the dataset
    
    Args:
        data (pd.DataFrame): Input dataframe
    """
    logger.info("\nDataset Info:")
    logger.info(f"Number of rows: {data.shape[0]}")
    logger.info(f"Number of columns: {data.shape[1]}")
    logger.info("\nColumn names:")
    for col in data.columns:
        logger.info(f"- {col}")
    logger.info("\nMissing values:")
    logger.info(data.isnull().sum())
