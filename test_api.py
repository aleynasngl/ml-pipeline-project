import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_api():
    # Test verisi oluştur
    train_data = {
        "data": [
            {
                "yas": 35,
                "gelir": 50000,
                "kredi_puani": 700,
                "cinsiyet": "E",
                "egitim_duzeyi": "Lisans",
                "hedef": 1
            },
            {
                "yas": 45,
                "gelir": 75000,
                "kredi_puani": 800,
                "cinsiyet": "K",
                "egitim_duzeyi": "Yüksek Lisans",
                "hedef": 0
            },
            {
                "yas": 28,
                "gelir": 45000,
                "kredi_puani": 650,
                "cinsiyet": "E",
                "egitim_duzeyi": "Lisans",
                "hedef": 1
            },
            {
                "yas": 52,
                "gelir": 90000,
                "kredi_puani": 850,
                "cinsiyet": "K",
                "egitim_duzeyi": "Doktora",
                "hedef": 0
            },
            {
                "yas": 31,
                "gelir": 55000,
                "kredi_puani": 720,
                "cinsiyet": "E",
                "egitim_duzeyi": "Yüksek Lisans",
                "hedef": 1
            },
            {
                "yas": 39,
                "gelir": 65000,
                "kredi_puani": 780,
                "cinsiyet": "K",
                "egitim_duzeyi": "Lisans",
                "hedef": 0
            }
        ],
        "numeric_features": ["yas", "gelir", "kredi_puani"],
        "categorical_features": ["cinsiyet", "egitim_duzeyi"],
        "target_column": "hedef"
    }

    # 1. Model Eğitimi
    print("\n1. Model Eğitimi:")
    train_response = requests.post(f"{BASE_URL}/train", json=train_data)
    print(f"Eğitim Yanıtı: {json.dumps(train_response.json(), indent=2, ensure_ascii=False)}")

    # 2. Tahmin
    print("\n2. Tahmin:")
    predict_data = {
        "data": [
            {
                "yas": 30,
                "gelir": 45000,
                "kredi_puani": 650,
                "cinsiyet": "E",
                "egitim_duzeyi": "Lisans"
            }
        ],
        "numeric_features": ["yas", "gelir", "kredi_puani"],
        "categorical_features": ["cinsiyet", "egitim_duzeyi"]
    }
    predict_response = requests.post(f"{BASE_URL}/predict", json=predict_data)
    print(f"Tahmin Yanıtı: {json.dumps(predict_response.json(), indent=2, ensure_ascii=False)}")

    # 3. Model Bilgileri
    print("\n3. Model Bilgileri:")
    model_info_response = requests.get(f"{BASE_URL}/model-info")
    print(f"Model Bilgileri: {json.dumps(model_info_response.json(), indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    test_api() 