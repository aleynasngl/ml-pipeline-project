from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import joblib
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
import os
import sys
from datetime import datetime
import io
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from data.preprocessing import preprocess_data, handle_missing_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Pipeline API",
    description="ML Pipeline API - Train ve Predict destekleniyor",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "ml-pipeline-models")
BEST_MODEL_PATH_IN_BUCKET = "models/best_model.joblib"
LOCAL_MODEL_CACHE = "/tmp/best_model.joblib"

storage_client = storage.Client()

class MLResponse(BaseModel):
    mode: str
    status: str
    message: str
    model_info: Optional[Dict[str, Any]] = None
    predictions: Optional[List[Any]] = None
    timestamp: str

def upload_model_to_gcs(local_path: str, bucket_path: str):
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Model {bucket_path} olarak GCS’ye yüklendi.")

def download_best_model_from_gcs():
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(BEST_MODEL_PATH_IN_BUCKET)
    if not os.path.exists(LOCAL_MODEL_CACHE):
        logger.info("En iyi model GCS’den indiriliyor...")
        blob.download_to_filename(LOCAL_MODEL_CACHE)
    else:
        logger.info("Model cache’den yükleniyor...")
    model_data = joblib.load(LOCAL_MODEL_CACHE)
    return model_data

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
            problem_type = "classification" if df[target_column].nunique() <= 10 else "regression"

            X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df, target_column)

            pipeline = MLPipeline(
                numeric_features=[],  # artık preprocessing yaptı
                categorical_features=[],
                target_column=target_column,
                model_name='auto',
                problem_type=problem_type
            )

            results = pipeline.auto_train_preprocessed(X_train_scaled, y_train, X_test_scaled, y_test)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_data = {
                'model': pipeline.best_model,
                'model_type': problem_type,
                'best_model': results['best_model'],
                'metrics': results['metrics'],
                'timestamp': timestamp,
                'target_column': target_column,
                'scaler': results['scaler'],
                'columns': df.drop(columns=[target_column]).columns.tolist()
            }

            joblib.dump(model_data, "/tmp/best_model.joblib")
            upload_model_to_gcs("/tmp/best_model.joblib", BEST_MODEL_PATH_IN_BUCKET)

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
            model_data = download_best_model_from_gcs()
            model = model_data['model']
            target_column = model_data['target_column']
            columns = model_data['columns']
            scaler = model_data['scaler']

            df = df[columns]  # sadece eğitimdeki feature'lar
            df = handle_missing_values(df)
            df_scaled = scaler.transform(df)

            predictions = model.predict(df_scaled)

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

@app.get("/")
async def root():
    return {
        "message": "ML Pipeline API'sine hoş geldiniz",
        "endpoints": {
            "/ml": "Train veya predict için .csv yükleyin",
            "/retrain": "retrain.py çağırır"
        }
    }
