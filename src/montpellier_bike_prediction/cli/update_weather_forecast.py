from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

from montpellier_bike_prediction.config import DEFAULT_LAT, DEFAULT_LON
from montpellier_bike_prediction.db_supabase import upsert_df

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_hourly_forecast_utc(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Fetch raw hourly forecast from Open-Meteo, in UTC.

    We request several days ahead (forecast_days=3) to be sure we cover
    "tomorrow" in UTC. The API returns:
      - 'time' as ISO string in UTC
      - hourly variables: temperature_2m, relative_humidity_2m,
        precipitation, wind_speed_10m
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "UTC",   # 👈 crucial: we work in pure UTC
        "forecast_days": 3,  # enough horizon to include tomorrow
    }

    print(f"➡ Calling Open-Meteo (UTC forecast) at lat={lat}, lon={lon}")
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected Open-Meteo response: keys={data.keys()}")

    h = data["hourly"]

    df = pd.DataFrame({
        "time_utc": pd.to_datetime(h["time"], utc=True),
        "temperature_2m": h.get("temperature_2m", []),
        "relative_humidity_2m": h.get("relative_humidity_2m", []),
        "precipitation": h.get("precipitation", []),
        "wind_speed_10m": h.get("wind_speed_10m", []),
    })

    return df


def keep_tomorrow_utc_midnight_to_midnight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exactly the 24 hours of *tomorrow in UTC*, from 00:00 to 23:00.

    - 'today' is defined in UTC
    - 'tomorrow' = today + 1 day
    - we build an hourly index from tomorrow 00:00:00 UTC to 23:00:00 UTC
    - we left-join Open-Meteo data on that index so we always have 24 rows.
    """
    now_utc = datetime.now(timezone.utc)
    tomorrow = (now_utc + timedelta(days=1)).date()

    print(f"📅 Now (UTC)         : {now_utc}")
    print(f"📅 Tomorrow (UTC day): {tomorrow}")

    # Build the 24 expected timestamps for tomorrow, in UTC
    hours = pd.date_range(
        start=datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, tzinfo=timezone.utc),
        periods=24,
        freq="H",
    )

    df = df.copy()
    # Ensure time_utc is tz-aware UTC
    df["time_utc"] = df["time_utc"].dt.tz_convert(timezone.utc)

    expected = pd.DataFrame({"time_utc": hours})
    df_merged = expected.merge(df, on="time_utc", how="left")

    print(f"✅ Forecast rows for tomorrow (UTC): {len(df_merged)} (should be 24)")
    return df_merged


def prepare_for_supabase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the forecast rows for insertion in Supabase table weather_forecast_hourly.

    The table schema is assumed to be something like:

        weather_forecast_hourly (
            timestamp_utc        timestamptz primary key,
            temperature_2m       double precision,
            relative_humidity_2m double precision,
            precipitation        double precision,
            wind_speed_10m       double precision,
            created_at           timestamptz default now()
        )

    We therefore:
      - rename time_utc -> timestamp_utc
      - format as ISO strings (without timezone suffix, Postgres will treat as UTC)
      - ensure standard float types
    """
    df = df.copy()

    # Rename and format timestamp for JSON → Supabase
    df["timestamp_utc"] = df["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    for col in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
        df[col] = df[col].astype(float)

    df_out = df[[
        "timestamp_utc",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    ]].drop_duplicates(subset=["timestamp_utc"])

    return df_out


def upsert_weather_forecast(df_forecast: pd.DataFrame, chunk_size: int = 500):
    """
    Upsert the forecast into the Supabase table `weather_forecast_hourly`.
    """
    total = len(df_forecast)
    if total == 0:
        print("⚠️ No forecast rows to upsert.")
        return

    print(f"🌦 Upserting {total} rows into weather_forecast_hourly (chunks of {chunk_size})")

    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk = df_forecast.iloc[start:end].copy()

        res = upsert_df("weather_forecast_hourly", chunk)
        print(f"  Chunk {i+1}/{n_chunks}: {res.get('count')} rows upserted")


def main():
    # 1) Fetch raw UTC forecast
    df_all = fetch_hourly_forecast_utc()

    # 2) Keep exactly "tomorrow" in UTC (00:00 → 23:00)
    df_tomorrow = keep_tomorrow_utc_midnight_to_midnight(df_all)

    print("\n🧾 Preview (time_utc + temp):")
    print(df_tomorrow[["time_utc", "temperature_2m"]].head())
    print(df_tomorrow[["time_utc", "temperature_2m"]].tail())

    # 3) Prepare for Supabase
    df_for_supabase = prepare_for_supabase(df_tomorrow)

    print(f"\n📦 Prepared for Supabase ({len(df_for_supabase)} rows):")
    print(df_for_supabase.head())

    # 4) Upload to Supabase
    upsert_weather_forecast(df_for_supabase)


if __name__ == "__main__":
    main()
