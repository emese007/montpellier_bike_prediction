# src/montpellier_bike_traffic/etl/weather_etl.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DEFAULT_LAT, DEFAULT_LON
from .weather_client import WeatherAPIClient


DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_weather_hourly_etl(
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> None:
    """
    ETL météo horaire : récupère l'historique Open-Meteo pour Montpellier
    entre start_date et end_date (YYYY-MM-DD).

    Sauvegarde un CSV/Parquet dans data/raw/ et data/processed/
    """
    import datetime as dt

    if end_date is None:
        today = dt.date.today().isoformat()
        end_date = today

    print(f"🌦 ETL météo horaire {start_date} → {end_date}")
    client = WeatherAPIClient()
    df = client.fetch_hourly_history(lat, lon, start_date, end_date)

    raw_path = DATA_RAW_DIR / "weather_hourly_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Météo horaire brute sauvegardée dans {raw_path}")

    # Ici tu pourras rajouter des colonnes, agrégations journalières, etc.
    processed_path = DATA_PROCESSED_DIR / "weather_hourly_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"✅ Météo horaire traitée sauvegardée dans {processed_path}")


if __name__ == "__main__":
    run_weather_hourly_etl()
