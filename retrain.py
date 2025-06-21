import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import logging
import sys

from google.cloud import storage

from src.pipeline import MLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.info(f"{source_blob_name} dosyası {bucket_name} bucket'ından indirildi → {destination_file_name}")
    except Exception as e:
        logger.error(f"GCS'den dosya indirme hatası: {str(e)}")
        raise

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        logger.info(f"Model GCS'ye yüklendi: gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        logger.error(f"Modeli GCS'ye yüklerken hata oluştu: {e}")
        raise

DATA_PATH = "data/latest_train.csv"
BUCKET_NAME = "mlpipeline-models"
GCS_FILE_NAME = "latest_train.csv"

os.makedirs("data", exist_ok=True)

try:
    download_from_gcs(BUCKET_NAME, GCS_FILE_NAME, DATA_PATH)
except Exception:
    logger.error("Veri indirilemedi. Retrain iptal edildi.")
    sys.exit(1)

if not os.path.exists(DATA_PATH):
    logger.error(f"{DATA_PATH} bulunamadı. Yeni veri yok, retrain iptal edildi.")
    sys.exit(0)

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    logger.error(f"Veri okuma hatası: {e}")
    sys.exit(1)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

possible_targets = [col for col in df.columns if 'diagnosis' in col.lower()]
target_column = possible_targets[0] if possible_targets else df.columns[-1]

if df[target_column].dtype == 'object':
    unique_values = df[target_column].unique()
    if len(unique_values) == 2:
        value_map = {val: i for i, val in enumerate(unique_values)}
        df[target_column] = df[target_column].map(value_map)

unique_vals = df[target_column].nunique()
if df[target_column].dtype == 'object' or unique_vals <= 10:
    problem_type = "classification"
else:
    problem_type = "regression"

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
    'categorical_features': categorical_features
}

model_dir = os.path.join("models", problem_type, results['best_model'])
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"model_{timestamp}.joblib")
joblib.dump(model_data, model_path)

logger.info(f"Model retrain edildi ve kaydedildi: {model_path}")

destination_blob_name = f"{problem_type}/{results['best_model']}/model_{timestamp}.joblib"
upload_to_gcs(BUCKET_NAME, model_path, destination_blob_name)
