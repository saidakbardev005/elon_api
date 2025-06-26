import os
import joblib
import numpy as np
import googlemaps
import pandas as pd
import warnings

from config import Config
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Warn messages suppression (optional)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Project root
base_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(base_dir, os.pardir))

def csv_dir():
    return os.path.join(project_root, "data")

# Paths for stored models
_MODEL_FILES = {
    "kmeans": os.path.join(project_root, "kmeans_model.pkl"),
    "scaler": os.path.join(project_root, "scaler.pkl"),
    "price": os.path.join(project_root, "price_predictor.pkl"),
}

# Caches for models and their mtimes
_models = {}
_mtimes = {}

def get_model(name: str):
    """
    Lazy-load or reload model if file has been updated.
    """
    path = _MODEL_FILES[name]
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        raise RuntimeError(f"Model file not found: {path}")
    if name not in _models or _mtimes.get(name, 0) < mtime:
        _models[name] = joblib.load(path)
        _mtimes[name] = mtime
    return _models[name]

# Google Maps Client
gmaps = googlemaps.Client(key=Config.GOOGLE_MAPS_API_KEY)

# Load CSV helper
def load_csv_data():
    data_dir    = csv_dir()
    df_price    = pd.read_csv(os.path.join(data_dir, "direction_prices.csv"))
    drivers_df  = pd.read_csv(os.path.join(data_dir, "driver_locations.csv"))
    my_autos    = pd.read_csv(os.path.join(data_dir, "my_autos.csv"))
    users       = pd.read_csv(os.path.join(data_dir, "users.csv"))
    return df_price, drivers_df, my_autos, users

# Geocoding helper
def get_coordinates(location: str):
    try:
        result = gmaps.geocode(location)
        if result:
            loc = result[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception:
        pass
    return None, None

# Price prediction
def predict_price(f_enc: int, t_enc: int) -> int:
    price_model = get_model("price")
    price_raw   = price_model.predict([[f_enc, t_enc]])[0]
    return int(f"{int(price_raw)}000")

# Best drivers finder with scaled KMeans
def find_best_drivers(lat: float, lon: float, weight: float, volume: float):
    df_price, drivers_df, my_autos, users = load_csv_data()

    # Merge driver info
    drivers_merged = drivers_df.merge(my_autos, on="user_id") \
                               .merge(users[["user_id","fullname","phone","status"]], on="user_id")

    # Ensure numeric and drop NaNs
    drivers_merged["transport_weight"] = pd.to_numeric(drivers_merged["transport_weight"], errors="coerce")
    drivers_merged["transport_volume"] = pd.to_numeric(drivers_merged["transport_volume"], errors="coerce")
    drivers_merged.dropna(subset=["transport_weight", "transport_volume"], inplace=True)

    # Extract weight/volume matrix
    arr = drivers_merged[["transport_weight", "transport_volume"]].values

    # 1) Scale features so weight & volume are comparable
    scaler_wv = StandardScaler().fit(arr)
    arr_scaled = scaler_wv.transform(arr)

    # 2) Dynamic KMeans on scaled data
    n_drivers  = arr_scaled.shape[0]
    n_clusters = min(4, n_drivers) if n_drivers > 1 else 1
    kmeans_local = KMeans(n_clusters=n_clusters, random_state=42)
    if n_drivers > 0:
        kmeans_local.fit(arr_scaled)

    # 3) Scale the input point and predict its cluster
    input_scaled = scaler_wv.transform(np.array([[weight, volume]]))
    cluster_id   = int(kmeans_local.predict(input_scaled)[0])

    # 4) Filter drivers in the same cluster
    drivers_merged["cluster_wv"] = kmeans_local.labels_
    same_cluster = drivers_merged[drivers_merged["cluster_wv"] == cluster_id].copy()
    if same_cluster.empty:
        return []

    # 5) Compute actual distances for ranking
    same_cluster["capacity_distance"] = np.sqrt(
        (same_cluster["transport_weight"] - weight) ** 2 +
        (same_cluster["transport_volume"] - volume) ** 2
    )
    same_cluster["distance_km"] = np.sqrt(
        (same_cluster["latitude"] - lat) ** 2 +
        (same_cluster["longitude"] - lon) ** 2
    ) * 111

    # 6) Sort by capacity_distance then geographic distance
    result = same_cluster.sort_values(
        by=["capacity_distance", "distance_km"],
        ascending=[True, True]
    ).head(5)[[
        "fullname", "phone", "transport_model",
        "transport_weight", "transport_volume", "distance_km"
    ]].to_dict(orient="records")

    return result
