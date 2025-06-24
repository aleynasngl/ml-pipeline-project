from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import joblib
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
import os
import io
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from data.preprocessing import preprocess_data, handle_missing_values

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="ML Pipeline API",
    description="ML Pipeline API - Train ve Predict destekleniyor",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Cloud Storage setup
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "ml-pipeline-models")
BEST_MODEL_PATH_IN_BUCKET = "models/best_model.joblib"
LOCAL_MODEL_CACHE = "/tmp/best_model.joblib"
storage_client = storage.Client()

# API response model
class MLResponse(BaseModel):
    mode: str
    status: str
    message: str
    model_info: Optional[Dict[str, Any]] = None
    predictions: Optional[List[Any]] = None
    timestamp: str

# Model upload
def upload_model_to_gcs(local_path: str, bucket_path: str):
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Model {bucket_path} olarak GCS’ye yüklendi.")

# Model yükleme (önce local, sonra GCS)
def get_best_model():
    if os.path.exists(LOCAL_MODEL_CACHE):
        logger.info("Local model bulundu, predict için o kullanılacak.")
        model_data = joblib.load(LOCAL_MODEL_CACHE)
        return model_data
    else:
        logger.info("Local model yok, GCS'den indiriliyor...")
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(BEST_MODEL_PATH_IN_BUCKET)
        blob.download_to_filename(LOCAL_MODEL_CACHE)
        model_data = joblib.load(LOCAL_MODEL_CACHE)
        return model_data

# ML Endpoint
@app.post("/ml", response_model=MLResponse)
async def ml_operation(
    file: UploadFile = File(...),
    mode: Literal["train", "predict"] = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.duplicated()]
        df = handle_missing_values(df)

        if mode == "train":
            from src.pipeline import MLPipeline

            target_column = df.columns[-1]

            # Problem tipi belirleme
            if df[target_column].dtype == 'object':
                problem_type = "classification"
            elif pd.api.types.is_numeric_dtype(df[target_column]):
                if df[target_column].nunique() <= 10 and all(df[target_column] % 1 == 0):
                    problem_type = "classification"
                else:
                    problem_type = "regression"
            else:
                problem_type = "classification"

            # Veri ön işleme
            X_train_scaled, X_test_scaled, y_train, y_test, scaler, used_columns = preprocess_data(df, target_column)

            pipeline = MLPipeline(
                numeric_features=[],  # varsayılan boş geçildi ama pipeline içinden alınabilir hale getirilebilir
                categorical_features=[],
                target_column=target_column,
                model_name='auto',
                problem_type=problem_type
            )

            # NOTE: auto_train_preprocessed yerine auto_train kullanılıyor artık
            df[target_column] = df[target_column].astype(y_train.dtype)
            combined = pd.concat([X_train_scaled, y_train], axis=1)
            results = pipeline.auto_train(combined)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_data = {
                'model': pipeline.best_model,
                'model_type': problem_type,
                'best_model': results['best_model'],
                'metrics': results['metrics'],
                'timestamp': timestamp,
                'target_column': target_column,
                'scaler': scaler,
                'columns': used_columns
            }

            joblib.dump(model_data, LOCAL_MODEL_CACHE)
            upload_model_to_gcs(LOCAL_MODEL_CACHE, BEST_MODEL_PATH_IN_BUCKET)

            return MLResponse(
                mode="train",
                status="success",
                message=f"Model başarıyla eğitildi: {results['best_model']}",
                model_info={
                    'model_type': problem_type,
                    'best_model': results['best_model'],
                    'metrics': results['metrics'],
                    'timestamp': timestamp
                },
                timestamp=timestamp
            )

        elif mode == "predict":
            model_data = get_best_model()
            model = model_data['model']
            target_column = model_data['target_column']
            columns = model_data['columns']
            scaler = model_data['scaler']

            logger.info(f"Gelen verideki kolonlar: {df.columns.tolist()}")
            logger.info(f"Beklenen kolonlar: {columns}")
            logger.info(f"Target kolon: {target_column}")

            if target_column.strip() in df.columns:
                df = df.drop(columns=[target_column.strip()])
                logger.info(f"Target kolonu '{target_column}' tahmin verisinden çıkarıldı.")

            df = handle_missing_values(df)

            # Kolon kontrolü
            missing = [col for col in columns if col not in df.columns]
            if missing:
                logger.error(f"Eksik kolonlar: {missing}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Tahmin verisinde eksik kolon(lar) var: {missing}"
                )

            df = df[columns]

            try:
                df_scaled = scaler.transform(df)
            except Exception as e:
                logger.error(f"Scaler transform hatası: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Veri ölçekleme hatası: {str(e)}")

            try:
                predictions = model.predict(df_scaled)
            except Exception as e:
                logger.error(f"Tahmin hatası: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Tahmin sırasında hata oluştu: {str(e)}")

            return MLResponse(
                mode="predict",
                status="success",
                message="Tahminler oluşturuldu.",
                predictions=predictions.tolist(),
                model_info={
                    'model_type': model_data['model_type'],
                    'best_model': model_data['best_model'],
                    'metrics': model_data['metrics'],
                    'timestamp': model_data['timestamp']
                },
                timestamp=datetime.now().isoformat()
            )

    except Exception as e:
        logger.error(f"ML işlemi hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ana sayfa
@app.get("/")
async def root():
    return {
        "message": "ML Pipeline API'sine hoş geldiniz",
        "endpoints": {
            "/ml": "Train veya predict için .csv yükleyin",
            "/retrain": "retrain.py çağırır"
        }
    }
