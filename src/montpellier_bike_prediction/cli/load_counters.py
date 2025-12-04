# src/montpellier_bike_prediction/cli/load_counters.py

from __future__ import annotations
import pandas as pd

from montpellier_bike_prediction.config import SELECTED_COUNTERS
from montpellier_bike_prediction.etl.bike_etl import BikeAPIClient
from montpellier_bike_prediction.db_supabase import upsert_df


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
    print("➡ Fetching counters from Montpellier API…")
    client = BikeAPIClient()
    df_all = client.fetch_all_counters()

    # Filter on selected counters
    df_sel = df_all[df_all["id"].isin(SELECTED_COUNTERS)].copy()

    # Extract readable name (if exists)
    if "name" in df_sel.columns:
        df_sel["name"] = df_sel["name"].apply(
            lambda x: x.get("value") if isinstance(x, dict) else None
        )
    else:
        df_sel["name"] = None

    # Extract latitude/longitude
    lats, lons = zip(*df_sel["location"].apply(extract_lat_lon))
    df_sel["lat"] = lats
    df_sel["lon"] = lons

    df_counters = df_sel[["id", "name", "lat", "lon"]]

    print("\n📍 Counters extracted:")
    print(df_counters)

    # Insert into Supabase
    res = upsert_df("counters", df_counters)
    print("\n✅ Upsert counters result:", res)


if __name__ == "__main__":
    run()
