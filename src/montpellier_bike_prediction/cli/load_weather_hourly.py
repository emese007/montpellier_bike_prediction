from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from montpellier_bike_prediction.db_supabase import upsert_df

# Racine du projet (un cran au-dessus de src/)
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_DIR / "data" / "processed" / "weather_hourly_processed.csv"


def load_weather_csv() -> pd.DataFrame:
    """
    Charge le CSV météo horaire et adapte les colonnes
    au schéma de la table weather_hourly :

      - timestamp_utc (timestamptz)
      - temperature_2m
      - relative_humidity_2m
      - precipitation
      - wind_speed_10m
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"📁 Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Colonnes météo:", df.columns.tolist())

    # Timestamp
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    elif "timestamp" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise ValueError("Colonne 'timestamp_utc' ou 'timestamp' manquante dans le CSV météo.")

    # Renommer pour coller au schéma SQL
    rename_map = {}
    if "temperature" in df.columns:
        rename_map["temperature"] = "temperature_2m"
    if "humidity" in df.columns:
        rename_map["humidity"] = "relative_humidity_2m"
    if "wind_speed" in df.columns:
        rename_map["wind_speed"] = "wind_speed_10m"

    df = df.rename(columns=rename_map)

    # S'assurer que les colonnes existent (sinon remplir avec None)
    for col in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
        if col not in df.columns:
            df[col] = None

    # Types simples (JSON-friendly)
    for col in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
        df[col] = df[col].astype(float)

    df_weather = df[["timestamp_utc", "temperature_2m", "relative_humidity_2m",
                     "precipitation", "wind_speed_10m"]].copy()

    # Unicité
    df_weather = df_weather.drop_duplicates(subset=["timestamp_utc"])

    # Conversion en string ISO pour JSON → Supabase (Postgres fait le cast en TIMESTAMPTZ)
    df_weather["timestamp_utc"] = df_weather["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return df_weather


def upsert_weather_hourly(df_weather: pd.DataFrame, chunk_size: int = 2000):
    total = len(df_weather)
    if total == 0:
        print("Aucune ligne météo à envoyer.")
        return

    print(f"🌦 Envoi de {total} lignes vers weather_hourly (paquets de {chunk_size})")

    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk = df_weather.iloc[start:end].copy()

        res = upsert_df("weather_hourly", chunk)
        print(f"  Chunk {i+1}/{n_chunks} : {res.get('count')} lignes upsertées")


def main():
    print(f"➡ Loading weather CSV from: {DATA_PATH}")
    df_weather = load_weather_csv()
    print(df_weather.head())
    upsert_weather_hourly(df_weather)


if __name__ == "__main__":
    main()
