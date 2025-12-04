# src/montpellier_bike_traffic/etl/holidays_client.py

from __future__ import annotations

from typing import Optional

import pandas as pd

from .base_client import BaseAPIClient


class HolidaysAPIClient(BaseAPIClient):
    """
    Client pour l'API jours fériés de l'État français.

    https://calendrier.api.gouv.fr/jours-feries/{annee}/{zone}.json
    """

    BASE_URL = "https://calendrier.api.gouv.fr"

    def fetch_year(self, year: int, zone: str = "metropole") -> pd.DataFrame:
        """
        Récupère les jours fériés pour une année et une zone donnés.
        Renvoie un DataFrame (date, name, zone, year).
        """
        data = self.get(f"/jours-feries/{zone}/{year}.json")

        records = [
            {"date": d, "name": name, "zone": zone, "year": year}
            for d, name in data.items()
        ]
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_range_for_training(self, start_year: int = 2023, zone: str = "metropole") -> pd.DataFrame:
        """
        Récupère les jours fériés de start_year à l'année courante incluse.
        Utile pour l'entraînement d'un modèle.
        """
        import datetime as dt

        current_year = dt.date.today().year
        frames = [self.fetch_year(y, zone=zone) for y in range(start_year, current_year + 1)]
        df_all = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
        return df_all
