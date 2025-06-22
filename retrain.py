import os
import sys
import logging
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from src.pipeline import MLPipeline
from datetime import datetime
import joblib
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "ml-pipeline-models-bucket")
storage_client = storage.Client()

def upload_model_to_gcs(local_path: str, bucket_path: str):
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Model {bucket_path} olarak bucket'a yüklendi.")

def download_dataset_from_kaggle(dataset_name: str, file_name: str, download_path: str):
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info(f"{file_name} Kaggle datasetinden indiriliyor...")

        result = api.dataset_download_file(dataset_name, file_name, path=download_path, force=True)
        logger.info(f"dataset_download_file fonksiyonu dönüş değeri: {result}")

        files = os.listdir(download_path)
        logger.info(f"{download_path} içeriği: {files}")

        full_path = os.path.join(download_path, file_name)
        if os.path.exists(full_path):
            logger.info(f"{file_name} bulundu: {full_path}")
            return full_path

        zip_files = [f for f in files if f.endswith(".zip")]
        if zip_files:
            zip_path = os.path.join(download_path, zip_files[0])
            logger.info(f"Zip dosyası bulundu: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_path)
            if os.path.exists(full_path):
                logger.info(f"{file_name} başarıyla açıldı: {full_path}")
                return full_path
            else:
                raise FileNotFoundError(f"{file_name} açıldı ama bulunamadı.")
        else:
            raise FileNotFoundError("Zip dosyası bulunamadı ve hedef dosya da yok.")

    except Exception as e:
        logger.error(f"Kaggle'dan dosya indirilemedi: {e}")
        sys.exit(1)

def save_best_model_locally(models_info):
    best_model = None
    best_metric = -float('inf')
    for info in models_info:
        acc = info['metrics'].get('accuracy', None)
        if acc is not None and acc > best_metric:
            best_metric = acc
            best_model = info
    return best_model

def main():
    dataset_name = "uciml/breast-cancer-wisconsin-data"
    file_name = "data.csv"
    csv_path = download_dataset_from_kaggle(dataset_name, file_name, DATA_DIR)

    df = pd.read_csv(csv_path)
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

    bucket_model_path = f"models/{problem_type}/{results['best_model']}/model_{timestamp}.joblib"
    upload_model_to_gcs(model_path, bucket_model_path)

    # Bucket'taki modelleri indirip kontrol et
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f"models/{problem_type}/"))
    models_info = []
    for blob in blobs:
        if blob.name.endswith(".joblib"):
            local_model_path = os.path.join("/tmp", os.path.basename(blob.name))
            blob.download_to_filename(local_model_path)
            model_info = joblib.load(local_model_path)
            models_info.append({'path': local_model_path, 'metrics': model_info.get('metrics', {})})

    best_model_info = save_best_model_locally(models_info)
    if best_model_info:
        best_blob = bucket.blob("models/best_model.joblib")
        best_blob.upload_from_filename(best_model_info['path'])
        logger.info(f"En iyi model güncellendi: models/best_model.joblib")

if __name__ == "__main__":
    main()
