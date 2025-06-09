import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from utils.model_utils import load_model
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess data for prediction
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        np.ndarray: Preprocessed data
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    return X_scaled

def predict(
    data_path: str,
    model_type: str,
    model_name: str,
    version: str
) -> dict:
    """
    Make predictions using a trained model
    
    Args:
        data_path (str): Path to data file
        model_type (str): Type of model ('classification' or 'regression')
        model_name (str): Name of the model
        version (str): Model version
        
    Returns:
        dict: Predictions and metadata
    """
    try:
        # Load data
        data = load_data(data_path)
        
        # Preprocess data
        X = preprocess_data(data)
        
        # Load model
        model = load_model(model_type, model_name, version)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Load model metrics
        metrics_path = os.path.join('models', model_type, model_name, f'{version}_metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Predictions completed successfully")
        
        return {
            'predictions': predictions.tolist(),
            'model_metrics': metrics,
            'model_type': model_type,
            'model_name': model_name,
            'version': version
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    data_path = "data/processed/test_data.csv"  # Update with your test data path
    model_type = "classification"  # or "regression"
    model_name = "logistic_regression"  # or any other model name from MODEL_MAP
    version = "20240315_123456"  # Update with your model version
    
    results = predict(data_path, model_type, model_name, version)
    print(f"Predictions completed. Results: {results}") 