import os
import sys
import logging
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from src.pipeline import MLPipeline
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset_from_kaggle(dataset_name: str, file_name: str, download_path: str):
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info(f"{file_name} Kaggle datasetinden indiriliyor...")

        # Dosyayı indir, fonksiyonun döndürdüğü yol
        result = api.dataset_download_file(dataset_name, file_name, path=download_path, force=True)
        logger.info(f"dataset_download_file fonksiyonu dönüş değeri: {result}")

        # İndirilen dizindeki dosyaları listele
        files = os.listdir(download_path)
        logger.info(f"{download_path} içeriği: {files}")

        full_path = os.path.join(download_path, file_name)
        if os.path.exists(full_path):
            logger.info(f"{file_name} bulundu: {full_path}")
            return full_path

        # Eğer zip dosyası varsa aç
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

def main():
    # İstediğin Kaggle dataset path ve csv dosya ismi
    dataset_name = "uciml/breast-cancer-wisconsin-data"
    file_name = "data.csv"

    # Dataset indir ve csv yolunu al
    csv_path = download_dataset_from_kaggle(dataset_name, file_name, DATA_DIR)

    # Veri oku
    df = pd.read_csv(csv_path)

    # Gereksiz Unnamed sütunları çıkar
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Hedef sütunu bul (diagnosis içeren varsa onu al)
    possible_targets = [col for col in df.columns if 'diagnosis' in col.lower()]
    target_column = possible_targets[0] if possible_targets else df.columns[-1]

    # Hedef ikili string ise sayısala çevir
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

if __name__ == "__main__":
    main()
