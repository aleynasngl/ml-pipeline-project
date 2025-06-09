from data.ingestion import load_local_data
from data.preprocessing import preprocess_data
from models.training import train_and_save_model

# 1. Veri yükle
df = load_local_data("data/data.csv")

# 2. Ön işleme
X, y = preprocess_data(df)

# 3. Model eğitimi
model, acc = train_and_save_model(X, y)