# data/ingestion.py

import pandas as pd

def load_local_data(path: str) -> pd.DataFrame:
    """
    CSV dosyasından veriyi yükler ve pandas DataFrame olarak döner.

    Parametre:
        path (str): CSV dosyasının yolu
    Dönüş:
        pd.DataFrame: Yüklenen veri
    """
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Dosya bulunamadı: {path}")
        return None
    except Exception as e:
        print(f"[ERROR] Veri yükleme sırasında hata: {e}")
        return None
