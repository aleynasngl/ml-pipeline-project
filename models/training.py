# models/training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pickle
import logging
import mlflow
import mlflow.sklearn
from typing import Tuple, Dict, Any

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Veri setini yükler"""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Veri seti başarıyla yüklendi: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Veri seti yüklenirken hata oluştu: {str(e)}")
        raise

def preprocess_data(data: pd.DataFrame, task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Veriyi ön işler ve task tipine göre hazırlar"""
    try:
        if task == "classification":
            X = data.iloc[:, :-1].values  # Son sütun hariç tüm özellikler
            y = data.iloc[:, -1].values   # Son sütun hedef değişken
        else:  # regression
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
        logger.info(f"Veri ön işleme tamamlandı. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Veri ön işleme sırasında hata oluştu: {str(e)}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, task: str) -> Tuple[Any, Dict[str, float]]:
    """Task tipine göre model eğitir"""
    try:
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Task tipine göre model seç
        if task == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred)
            }
        else:  # regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }
        
        logger.info(f"{task} modeli eğitildi. Metrikler: {metrics}")
        return model, metrics
    except Exception as e:
        logger.error(f"Model eğitimi sırasında hata oluştu: {str(e)}")
        raise

def save_model(model: Any, task: str, metrics: Dict[str, float]):
    """Modeli ve metrikleri kaydeder"""
    try:
        # Modeli kaydet
        model_path = f"models/{task}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # MLflow ile logla
        with mlflow.start_run(run_name=f"{task}_model"):
            mlflow.log_params({
                "task": task,
                "model_type": type(model).__name__
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, f"{task}_model")
        
        logger.info(f"Model ve metrikler başarıyla kaydedildi: {model_path}")
    except Exception as e:
        logger.error(f"Model kaydedilirken hata oluştu: {str(e)}")
        raise

def main():
    """Ana eğitim pipeline'ı"""
    try:
        # MLflow tracking server'ı başlat
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Her task için model eğit
        tasks = ["classification", "regression"]
        for task in tasks:
            # Veri setini yükle
            data = load_data(f"data/{task}_data.csv")
            
            # Veriyi ön işle
            X, y = preprocess_data(data, task)
            
            # Modeli eğit
            model, metrics = train_model(X, y, task)
            
            # Modeli kaydet
            save_model(model, task, metrics)
            
        logger.info("Tüm modeller başarıyla eğitildi ve kaydedildi")
    except Exception as e:
        logger.error(f"Eğitim pipeline'ı sırasında hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    main()
