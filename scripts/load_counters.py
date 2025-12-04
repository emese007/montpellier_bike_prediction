# src/montpellier_bike_traffic/scripts/load_counters.py

from __future__ import annotations

import pandas as pd

from montpellier_bike_prediction.src.montpellier_bike_prediction.config import SELECTED_COUNTERS
from montpellier_bike_prediction.db_supabase import upsert_df
from montpellier_bike_prediction.etl.bike_client import BikeAPIClient


def extract_lat_lon(location_obj):
    """
    Extrait (lat, lon) d'un champ NGSI location.
    location.value.coordinates = [lon, lat]
    """
    if (
        isinstance(location_obj, dict)
        and "value" in location_obj
        and isinstance(location_obj["value"], dict)
        and "coordinates" in location_obj["value"]
    ):
        coords = location_obj["value"]["coordinates"]
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            return lat, lon
    return None, None


def run():
    client = BikeAPIClient()
    df_all = client.fetch_all_counters()

    # On garde uniquement les 10 sélectionnés
    df_sel = df_all[df_all["id"].isin(SELECTED_COUNTERS)].copy()

    # Nom lisible si dispo (sinon on laisse None pour l'instant)
    if "name" in df_sel.columns:
        df_sel["name_col"] = df_sel["name"].apply(
            lambda x: x.get("value") if isinstance(x, dict) else None
        )
    else:
        df_sel["name_col"] = None

    # Lat / lon
    lats, lons = zip(*df_sel["location"].apply(extract_lat_lon))
    df_sel["lat"] = lats
    df_sel["lon"] = lons

    df_counters = df_sel[["id", "name_col", "lat", "lon"]].rename(
        columns={"name_col": "name"}
    )

    print(df_counters)

    res = upsert_df("counters", df_counters)
    print("Upsert counters result:", res)


if __name__ == "__main__":
    run()
