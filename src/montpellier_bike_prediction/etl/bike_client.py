# src/montpellier_bike_traffic/etl/bike_client.py

from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from .base_client import BaseAPIClient


class BikeAPIClient(BaseAPIClient):
    """
    Client pour l'API EcoCounter de Montpellier 3M.

    Base URL : https://portail-api-data.montpellier3m.fr
    """

    BASE_URL = "https://portail-api-data.montpellier3m.fr"

    def fetch_all_counters(self, limit: int = 1000) -> pd.DataFrame:
        """
        Récupère la liste de tous les compteurs EcoCounter via /ecocounter avec pagination.
        Renvoie un DataFrame brut.
        """
        offset = 0
        all_results: list[dict] = []

        while True:
            data = self.get(
                "/ecocounter",
                params={"limit": limit, "offset": offset},
                timeout=30,
            )
            if not isinstance(data, list) or len(data) == 0:
                break
            all_results.extend(data)
            offset += limit

        return pd.DataFrame(all_results)

    def fetch_timeseries(
        self,
        counter_id: str,
        start: dt.datetime,
        end: dt.datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Récupère la série temporelle 'intensity' pour un compteur donné
        entre deux datetimes UTC (start, end), via :

            /ecocounter_timeseries/{counter_id}/attrs/intensity

        Renvoie un DataFrame (timestamp_utc, intensity, ecocounter_id)
        ou None s'il n'y a pas de données.
        """
        params = {
            "fromDate": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "toDate": end.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            ts = self.get(
                f"/ecocounter_timeseries/{counter_id}/attrs/intensity",
                params=params,
                timeout=60,
            )
        except Exception as e:
            print(f"⚠️ Erreur API pour {counter_id} ({params}): {e}")
            return None

        if "index" not in ts or "values" not in ts or len(ts["index"]) == 0:
            return None

        df = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(ts["index"], utc=True),
                "intensity": ts["values"],
                "ecocounter_id": ts.get("entityId", counter_id),
            }
        )
        return df
