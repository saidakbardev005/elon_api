import os
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import Config

ROOT = os.path.dirname(__file__)

def load_all_data():
    # 1) CSVlarni o‘qing
    locs = pd.read_csv(os.path.join(ROOT, "data/driver_locations.csv"))
    autos = pd.read_csv(os.path.join(ROOT, "data/my_autos.csv"))
    users = pd.read_csv(os.path.join(ROOT, "data/users.csv"))
    prices = pd.read_csv(os.path.join(ROOT, "data/direction_prices.csv"))

    # 2) Shahar nomlarini kodlash uchun LabelEncoder
    le = LabelEncoder()
    prices["frm_enc"] = le.fit_transform(prices["From"].astype(str))
    prices["to_enc"]  = le.transform(prices["To"].astype(str))

    # 3) Haydovchi + avtomobil + user ma’lumotlarini birlashtiring
    df = locs.merge(autos, on="user_id") \
             .merge(users, on="user_id")
    # Shu df’da: latitude, longitude, transport_weight, transport_volume…

    return df, prices, le

def train_and_save_models():
    df_locs, df_prices, le = load_all_data()

    # — Cluster modeli (weight/volume bo‘yicha)
    X_cluster = df_locs[["transport_weight", "transport_volume"]].values
    n_drivers = len(X_cluster)
    n_clusters = min(Config.N_CLUSTERS, n_drivers) if n_drivers>1 else 1
    km = KMeans(n_clusters=n_clusters, random_state=0)
    if n_drivers > 0:
        km.fit(X_cluster)
    joblib.dump(km, os.path.join(ROOT, "kmeans_model.pkl"))

    # — Price predict modeli (encoded cities → narx)
    scaler = StandardScaler()
    X_price = scaler.fit_transform(df_prices[["frm_enc", "to_enc"]])
    y_price = df_prices["Price"]
    price_model = SGDRegressor(max_iter=1000, tol=1e-3)
    price_model.fit(X_price, y_price)

    joblib.dump(scaler,      os.path.join(ROOT, "scaler.pkl"))
    joblib.dump(price_model, os.path.join(ROOT, "price_predictor.pkl"))
    joblib.dump(le,          os.path.join(ROOT, "label_encoder.pkl"))

if __name__ == "__main__":
    train_and_save_models()
