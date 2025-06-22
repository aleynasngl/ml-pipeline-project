# data/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise


def handle_missing_values(data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        data (pd.DataFrame): Input dataframe
        strategy (str): Strategy to handle missing values ('mean', 'median', 'mode')
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    try:
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'mode':
            return data.fillna(data.mode().iloc[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise
