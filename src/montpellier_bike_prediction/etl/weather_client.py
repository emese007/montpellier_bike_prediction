# src/montpellier_bike_traffic/etl/weather_client.py

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests


class WeatherAPIClient:
    """
    Client pour l'API Open-Meteo (historical + forecast).

    On utilise l'endpoint "archive" pour l'historique :
    https://archive-api.open-meteo.com/v1/archive
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def fetch_hourly_history(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Récupère les données météo horaires entre start_date et end_date (YYYY-MM-DD).

        Variables :
        - temperature_2m
        - relative_humidity_2m
        - precipitation
        - wind_speed_10m
        - wind_gusts_10m
        - cloudcover
        - shortwave_radiation
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "cloudcover",
                    "shortwave_radiation",
                ]
            ),
            "timezone": "UTC",
        }

        resp = self.session.get(self.BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        df = pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True)})
        for key in [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
        ]:
            df[key] = hourly.get(key, [None] * len(times))

        return df
