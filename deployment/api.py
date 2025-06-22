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

        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logger.info(f"Unnamed sütunlar kaldırıldı: {unnamed_cols}")

        if mode == "train":
            from src.pipeline import MLPipeline

            possible_targets = [col for col in df.columns if 'diagnosis' in col.lower()]
            target_column = possible_targets[0] if possible_targets else df.columns[-1]

            value_maps = {}
            if df[target_column].dtype == 'object':
                unique_values = df[target_column].unique()
                if len(unique_values) == 2:
                    value_map = {val: i for i, val in enumerate(unique_values)}
                    df[target_column] = df[target_column].map(value_map)
                    logger.info(f"Hedef sütun sayısala çevrildi: {value_map}")

            unique_vals = df[target_column].nunique()
            if df[target_column].dtype == 'object' or unique_vals <= 10:
                problem_type = "classification"
            else:
                problem_type = "regression"

            numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if target_column in numeric_features:
                numeric_features.remove(target_column)
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()

            for col in categorical_features:
                unique_vals = df[col].unique()
                val_map = {val: i for i, val in enumerate(unique_vals)}
                value_maps[col] = val_map
                df[col] = df[col].map(val_map)

            pipeline = MLPipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                target_column=target_column,
                model_name='auto',
                problem_type=problem_type
            )

            results = pipeline.auto_train(df)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_data = {
                'model': pipeline.best_model,
                'model_type': problem_type,
                'best_model': results['best_model'],
                'metrics': results['metrics'],
                'feature_importance': results['feature_importance'],
                'timestamp': timestamp,
                'target_column': target_column,
                'numeric_features': numeric_features,
                'categorical_features': categorical_features,
                'value_maps': value_maps
            }

            model_dir = os.path.join("models", problem_type, results['best_model'])
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_{timestamp}.joblib")
            joblib.dump(model_data, model_path)
            logger.info(f"Model kaydedildi: {model_path}")

            joblib.dump(model_data, "/tmp/best_model.joblib")
            upload_model_to_gcs("/tmp/best_model.joblib", BEST_MODEL_PATH_IN_BUCKET)

            serializable_model_info = {
                'model_type': problem_type,
                'best_model': results['best_model'],
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
                message=f"Model başarıyla eğitildi ve kaydedildi. En iyi model: {results['best_model']}",
                model_info=serializable_model_info,
                timestamp=timestamp
            )

        elif mode == "predict":
            model_data = download_best_model_from_gcs()
            model = model_data['model']

            value_maps = model_data.get('value_maps', {})
            for col, val_map in value_maps.items():
                if col in df.columns:
                    df[col] = df[col].map(val_map)

            feature_columns = model_data['numeric_features'] + model_data['categorical_features']
            df = df[feature_columns]

            predictions = model.predict(df)

            return MLResponse(
                mode="predict",
                status="success",
                message="Tahminler başarıyla oluşturuldu.",
                predictions=predictions.tolist(),
                model_info={
                    'model_type': model_data['model_type'],
                    'best_model': model_data['best_model'],
                    'metrics': model_data['metrics'],
                    'feature_importance': model_data['feature_importance'],
                    'timestamp': model_data['timestamp'],
                    'target_column': model_data['target_column'],
                    'numeric_features': model_data['numeric_features'],
                    'categorical_features': model_data['categorical_features']
                },
                timestamp=datetime.now().isoformat()
            )
        else:
            return MLResponse(
                mode=mode,
                status="error",
                message="Geçersiz mode değeri, sadece 'train' veya 'predict' olabilir.",
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"ML işlemi sırasında hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    def run_retrain():
        import subprocess
        try:
            retrain_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrain.py")
            result = subprocess.run(
                [sys.executable, retrain_script],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Retrain scripti başarıyla çalıştı:\n" + result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Retrain scripti hata verdi:\n{e.stderr}")

    background_tasks.add_task(run_retrain)
    return {"status": "success", "message": "Retrain işlemi arka planda başlatıldı."}

@app.get("/")
async def root():
    return {
        "message": "ML Pipeline API'sine hoş geldiniz",
        "endpoints": {
            "/ml": "CSV dosyası yükleyip train veya predict işlemi yapmak için (mode=train veya mode=predict)",
            "/retrain": "Manuel retrain işlemi başlatmak için"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
