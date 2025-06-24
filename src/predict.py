import pandas as pd
import numpy as np
import logging
import json
import os
from sklearn.preprocessing import StandardScaler
from utils.model_utils import load_model

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data: pd.DataFrame, expected_columns: list) -> np.ndarray:
    """
    Veriyi feature sırasına göre düzenle ve ölçekle

    Args:
        data (pd.DataFrame): Input data
        expected_columns (list): Train sırasında kullanılan kolonlar

    Returns:
        np.ndarray: Scaled feature array
    """
    logger.info(f"Expected columns: {expected_columns}")
    logger.info(f"Incoming columns: {list(data.columns)}")

    # Target varsa çıkar
    if 'target' in data.columns:
        data = data.drop(columns=['target'])

    # Eksik kolon kontrolü
    missing = [col for col in expected_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Veride eksik kolon(lar) var: {missing}")

    # Kolon sıralamasını sabitle
    data = data[expected_columns]

    # Ölçekleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)  # Not: Scaler train sırasında kaydedilmediyse yeniden fit ediliyor

    return X_scaled

def predict(
    data_path: str,
    model_type: str,
    model_name: str,
    version: str
) -> dict:
    try:
        # Veri ve model yolları
        base_path = os.path.join('models', model_type, model_name)
        columns_path = os.path.join(base_path, f"{version}_columns.json")
        metrics_path = os.path.join(base_path, f"{version}_metrics.json")

        # Load data
        data = load_data(data_path)

        # Load expected feature columns
        with open(columns_path, 'r') as f:
            expected_columns = json.load(f)

        # Preprocess
        X = preprocess_data(data, expected_columns)

        # Load model
        model = load_model(model_type, model_name, version)

        # Predict
        predictions = model.predict(X)

        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        logger.info("Prediction completed successfully")

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
    data_path = "data/processed/test_data.csv"  # ⚠️ Güncelle
    model_type = "classification"              # "regression" olabilir
    model_name = "logistic_regression"         # MODEL_MAP içinde tanımlı olmalı
    version = "20240624_123456"                # Eğitimde üretilen versiyon

    results = predict(data_path, model_type, model_name, version)
    print(f"Predictions completed. Results: {results}")
