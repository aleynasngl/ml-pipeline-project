import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from utils.model_utils import create_model, evaluate_model, save_model
import os
import json
from datetime import datetime

# Configure logging
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

def preprocess_data(data: pd.DataFrame, target_column: str):
    """
    Split ve scale işlemi yapar, aynı zamanda feature isimlerini döner.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)

def train_model(
    data_path: str,
    target_column: str,
    model_type: str,
    model_name: str,
    version: str = None
) -> dict:
    try:
        data = load_data(data_path)
        X_train, X_test, y_train, y_test, scaler, used_columns = preprocess_data(data, target_column)

        model = create_model(model_type, model_name)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, model_type)

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = save_model(model, model_type, model_name, version)

        # Save metrics
        metrics_path = os.path.join('models', model_type, model_name, f'{version}_metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # ✅ Save used feature columns
        columns_path = os.path.join('models', model_type, model_name, f'{version}_columns.json')
        with open(columns_path, 'w') as f:
            json.dump(used_columns, f, indent=4)

        logger.info(f"Model training completed. Metrics: {metrics}")
        logger.info(f"Used feature columns saved to: {columns_path}")

        return {
            'metrics': metrics,
            'model_path': model_path,
            'metrics_path': metrics_path,
            'columns_path': columns_path
        }

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    data_path = "data/processed/your_data.csv"  # ← Güncelle
    target_column = "target"  # ← Güncelle
    model_type = "classification"  # or "regression"
    model_name = "logistic_regression"  # or another from MODEL_MAP
    results = train_model(data_path, target_column, model_type, model_name)
    print(f"Training completed. Results: {results}")
