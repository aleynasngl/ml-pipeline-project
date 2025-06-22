# --- SENİN ORJİNAL DOSYANIN İÇERİĞİ BAŞI ---
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import joblib
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
import os
import sys
from datetime import datetime
import mlflow
import io
from fastapi.middleware.cors import CORSMiddleware
import json

from google.cloud import storage

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.pipeline import MLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Pipeline API",
    description="Otomatik ML Pipeline API'si - CSV dosyası yükleyip train veya predict yapın",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MLResponse(BaseModel):
    mode: str
    status: str
    message: str
    model_info: Optional[Dict[str, Any]] = None
    predictions: Optional[List[Any]] = None
    timestamp: str

MODELS_DIR = os.path.join(project_root, "models")
CLASSIFICATION_DIR = os.path.join(MODELS_DIR, "classification")
REGRESSION_DIR = os.path.join(MODELS_DIR, "regression")

GCS_BUCKET_NAME = "mlpipeline-models"

def upload_to_gcs(bucket_name: str, destination_blob_name: str, data: bytes):
    logger.info(f"GCS yüklemesi başlıyor: bucket={bucket_name}, blob={destination_blob_name}")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(data)
        logger.info(f"GCS yüklemesi başarılı: {destination_blob_name}")
    except Exception as e:
        logger.error(f"Dosya GCS'ye yüklenirken hata: {str(e)}")
        raise

def determine_problem_type(df: pd.DataFrame, target_column: str) -> str:
    target = df[target_column]
    unique_values = target.nunique()

    if target.dtype == 'object' or unique_values <= 10:
        return 'classification'
    elif target.dtype in ['int64', 'float64'] and unique_values > 10:
        return 'regression'
    else:
        return 'classification'

@app.post("/ml", response_model=MLResponse)
async def ml_operation(
    file: UploadFile = File(...),
    mode: Literal["train", "predict"] = Form(...)
):
    try:
        contents = await file.read()

        # Timestamp ve dosya ismini oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.splitext(file.filename)[0].replace(" ", "_")
        gcs_filename = f"datasets/{original_filename}_{timestamp}.csv"

        logger.info(f"upload_to_gcs fonksiyonu çağrılacak, dosya adı: {gcs_filename}")
        upload_to_gcs(GCS_BUCKET_NAME, gcs_filename, contents)

        df = pd.read_csv(io.BytesIO(contents))

        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logger.info(f"Unnamed sütunlar kaldırıldı: {unnamed_cols}")

        possible_targets = [col for col in df.columns if 'diagnosis' in col.lower()]
        if possible_targets:
            target_column = possible_targets[0]
            logger.info(f"Anlamlı hedef sütun bulundu: {target_column}")
        else:
            target_column = df.columns[-1]
            logger.info(f"Son sütun hedef olarak belirlendi: {target_column}")

        if df[target_column].dtype == 'object':
            unique_values = df[target_column].unique()
            if len(unique_values) == 2:
                value_map = {val: i for i, val in enumerate(unique_values)}
                df[target_column] = df[target_column].map(value_map)
                logger.info(f"Hedef sütun sayısala çevrildi: {value_map}")

        problem_type = determine_problem_type(df, target_column)
        logger.info(f"Problem tipi belirlendi: {problem_type}")

        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        pipeline = MLPipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_column=target_column,
            model_name='auto',
            problem_type=problem_type
        )

        if mode == "train":
            results = pipeline.auto_train(df)

            best_model_type = results['best_model']
            model_dir = os.path.join('models', problem_type, best_model_type)
            os.makedirs(model_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_data = {
                'model': pipeline.best_model,
                'model_type': problem_type,
                'best_model': best_model_type,
                'metrics': results['metrics'],
                'feature_importance': results['feature_importance'],
                'timestamp': timestamp,
                'target_column': target_column,
                'numeric_features': numeric_features,
                'categorical_features': categorical_features
            }

            model_path = os.path.join(model_dir, f"model_{timestamp}.joblib")
            joblib.dump(model_data, model_path)

            logger.info(f"En iyi model: {best_model_type}")
            logger.info(f"Model kaydedildi: {model_path}")

            serializable_model_info = {
                'model_type': problem_type,
                'best_model': best_model_type,
                'metrics': results['metrics'],
                'feature_importance': results['feature_importance'].tolist() if isinstance(results['feature_importance'], np.ndarray) else results['feature_importance'],
                'timestamp': timestamp,
                'target_column': target_column,
                'numeric_features': numeric_features,
                'categorical_features': categorical_features
            }

            return MLResponse(
                mode="train",
                status="success",
                message=f"Model başarıyla eğitildi ve kaydedildi. En iyi model: {best_model_type}",
                model_info=serializable_model_info,
                timestamp=timestamp
            )

        else:
            latest_model = None
            latest_info = None
            latest_timestamp = None

            for root, dirs, files in os.walk('models'):
                for file in files:
                    if file.endswith('.joblib'):
                        file_path = os.path.join(root, file)
                        model_data = joblib.load(file_path)
                        if latest_timestamp is None or model_data['timestamp'] > latest_timestamp:
                            latest_timestamp = model_data['timestamp']
                            latest_info = {
                                'model_type': model_data['model_type'],
                                'best_model': model_data['best_model'],
                                'metrics': model_data['metrics'],
                                'feature_importance': model_data['feature_importance'].tolist() if isinstance(model_data['feature_importance'], np.ndarray) else model_data['feature_importance'],
                                'timestamp': model_data['timestamp'],
                                'target_column': model_data['target_column'],
                                'numeric_features': model_data['numeric_features'],
                                'categorical_features': model_data['categorical_features']
                            }
                            latest_model = model_data['model']

            if latest_model is None:
                raise HTTPException(status_code=400, detail="Eğitilmiş model bulunamadı. Önce model eğitimi yapın.")

            predictions = latest_model.predict(df)

            return MLResponse(
                mode="predict",
                status="success",
                message="Tahminler başarıyla oluşturuldu",
                predictions=predictions.tolist(),
                model_info=latest_info,
                timestamp=datetime.now().isoformat()
            )

    except Exception as e:
        logger.error(f"ML işlemi sırasında hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "ML Pipeline API'sine hoş geldiniz",
        "endpoints": {
            "/ml": "CSV dosyası yükleyip train veya predict işlemi yapmak için"
        }
    }

# ----------- BURAYA ALTTAKİ KISMI EKLE -----------

import subprocess
from fastapi import BackgroundTasks

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Arka planda retrain.py scriptini çalıştırır.
    """
    def run_retrain():
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(project_root, "retrain.py")],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Retrain scripti başarıyla çalıştı:\n" + result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Retrain scripti hata verdi:\n{e.stderr}")

    background_tasks.add_task(run_retrain)

    return {"status": "success", "message": "Retrain işlemi arka planda başlatıldı."}

# ---------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
