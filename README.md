## Proje Zaman Çizelgesi (Güncel)

| Hafta   | Görev                                    | Durum      |
|---------|-------------------------------------------|------------|
| 1-2     | Ortam kurulumu, mimari tasarım            | TAMAMLANDI |
| 3-4     | Veri ingestion, model eğitimi, API kurulumu | SONRAKİ ADIM |
| 5-6     | Dockerization, Cloud deploy, MLflow entegrasyon | - |
| 7       | İlk prototip demo ve rapor hazırlığı      | - |

# ML Pipeline API
models/training.py dosyasındaki kodu senin için satır satır açıklıyorum. Bu dosya, model eğitim sürecinin tam kalbidir – modeli eğitir, performansını ölçer, kaydeder ve MLflow ile loglar. 
S
## API Endpoint'leri

### POST /predict
- **Amaç:** Gönderilen veriden tahmin döndürür.
- **Input:** JSON formatında veri ([feature1, feature2, ...])
- **Output:** Tahmin sonucu (JSON)

### POST /retrain (Gelişmiş aşama)
- **Amaç:** Modelin performansı düştüğünde retrain tetikler.
- **Input:** (Opsiyonel parametreler)
- **Output:** Başarılı retrain mesajı

## Performans Metrikleri

- **Model Doğruluk (Accuracy)**
- **F1-Score**
- **Recall & Precision**
- **API Yanıt Süresi (ms)**
- **CPU, Memory Kullanımı (CloudWatch)**

## İzleme

- **MLflow Tracking Server:** Model geçmişini, parametreleri, metrikleri ve versiyonları loglar.
- **CloudWatch:** API ve sistem sağlık kontrollerini sağlar.

Bu proje, Iris çiçeği sınıflandırması için kapsamlı bir makine öğrenmesi pipeline'ı içerir. Proje, veri işleme, model eğitimi, model kaydı, performans izleme ve API deployment aşamalarını içerir.

## 🚀 Özellikler

- **Veri İşleme**: Veri yükleme, temizleme ve ön işleme
- **Model Eğitimi**: RandomForest sınıflandırıcı ile model eğitimi
- **MLflow Entegrasyonu**: Model ve metriklerin izlenmesi
- **FastAPI Servisi**: Model deployment ve tahmin API'si
- **Otomatik Testler**: Pipeline'ın doğru çalıştığını doğrulama

## 📋 Gereksinimler

```bash
pandas>=1.3.0
numpy>=1.19.0
scikit-learn>=0.24.0
mlflow>=2.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
```

## 🛠️ Kurulum

1. Projeyi klonlayın:
```bash
git clone <repo-url>
cd ml-pipeline-cloud
```

2. Sanal ortam oluşturun ve aktive edin:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

## 📁 Proje Yapısı

```
ml-pipeline-cloud/
├── data/                  # Veri dosyaları
│   └── iris_data.csv     # Iris veri seti
├── models/               # Model dosyaları
│   ├── model.pkl        # Eğitilmiş model
│   └── training.py      # Model eğitim kodu
├── deployment/          # Deployment dosyaları
│   └── api.py          # FastAPI servisi
├── tests/              # Test dosyaları
│   └── test_pipeline.py # Pipeline testleri
├── mlruns/             # MLflow deneyleri
├── requirements.txt    # Proje bağımlılıkları
└── README.md          # Proje dokümantasyonu
```

## 🚀 Kullanım

### 1. MLflow Server'ı Başlatma

```bash
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri file:./mlruns
```

### 2. Model Eğitimi ve Test

```bash
python test_pipeline.py
```

Bu komut:
- Iris veri setini yükler
- Veriyi ön işler
- Modeli eğitir
- Performans metriklerini hesaplar
- Modeli MLflow'a kaydeder

### 3. API Servisini Başlatma

```bash
python deployment/api.py
```

API şu endpoint'leri sunar:
- `GET /`: API sağlık kontrolü
- `POST /predict`: Iris çiçeği sınıflandırması

### 4. API'yi Test Etme

#### Swagger UI ile Test
1. Tarayıcıda `http://localhost:8000/docs` adresini açın
2. `/predict` endpoint'ini seçin
3. "Try it out" butonuna tıklayın
4. Örnek veri girin:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### cURL ile Test
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 📊 MLflow Arayüzü

MLflow arayüzüne `http://127.0.0.1:5000` adresinden erişebilirsiniz. Burada:
- Deneyleri görüntüleyebilirsiniz
- Model performans metriklerini inceleyebilirsiniz
- Model versiyonlarını takip edebilirsiniz

## 🔍 Model Performansı
Model aşağıdaki metriklerle değerlendirilir:
- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-score

CI/CD pipeline test - Thu Jun 19 19:12:50 +03 2025
