from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd
from prophet import Prophet

from montpellier_bike_prediction.config import SELECTED_COUNTERS
from montpellier_bike_prediction.db_supabase import get_supabase_client, upsert_df


# ---------- LOADERS FROM SUPABASE ----------


def load_bike_history(counter_id: str) -> pd.DataFrame:
    """
    Load hourly bike history for one counter.
    Table: bike_hourly(counter_id, timestamp_utc, intensity)
    """
    client = get_supabase_client()
    resp = (
        client.table("bike_hourly")
        .select("counter_id, timestamp_utc, intensity")
        .eq("counter_id", counter_id)
        .order("timestamp_utc")
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise ValueError(f"No bike data in bike_hourly for {counter_id}")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def load_weather_history() -> pd.DataFrame:
    """
    Load full hourly weather history.
    Table: weather_hourly(timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m)
    """
    client = get_supabase_client()
    resp = (
        client.table("weather_hourly")
        .select("timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m")
        .order("timestamp_utc")
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise ValueError("No data in weather_hourly")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def load_holidays() -> pd.DataFrame:
    """
    Load holidays calendar.
    Table: holidays(date, name, year)
    """
    client = get_supabase_client()
    resp = client.table("holidays").select("date, name, year").execute()
    rows = resp.data or []
    if not rows:
        raise ValueError("No data in holidays")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_weather_forecast_for_tomorrow() -> pd.DataFrame:
    """
    Load hourly weather forecast from weather_forecast_hourly,
    and keep only tomorrow (UTC).
    """
    client = get_supabase_client()
    resp = (
        client.table("weather_forecast_hourly")
        .select("timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m")
        .order("timestamp_utc")
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise ValueError("No data in weather_forecast_hourly")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    now_utc = datetime.now(timezone.utc)
    tomorrow = (now_utc + timedelta(days=1)).date()
    df["date"] = df["timestamp_utc"].dt.date
    df = df[df["date"] == tomorrow].copy()

    if df.empty:
        raise ValueError("No forecast rows for tomorrow in weather_forecast_hourly")

    return df


# ---------- TRAINING DATASET ----------


def build_training_dataframe(
    counter_id: str,
    df_bike: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_holidays: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build training dataframe for Prophet for one counter:
      - ds : timestamp_utc (UTC)
      - y  : intensity
      - regressors: temperature_2m, relative_humidity_2m,
                    precipitation, wind_speed_10m,
                    is_holiday, dow, hour
    """
    # Join bike + weather on timestamp_utc
    df = pd.merge(
        df_bike,
        df_weather,
        on="timestamp_utc",
        how="left",
    )

    # Join with holidays
    df["date"] = df["timestamp_utc"].dt.date
    df = df.merge(
        df_holidays[["date", "name"]],
        on="date",
        how="left",
        suffixes=("", "_holiday"),
    )
    df["is_holiday"] = df["name"].notna().astype(int)

    # Time features
    df["dow"] = df["timestamp_utc"].dt.weekday
    df["hour"] = df["timestamp_utc"].dt.hour

    # Prophet columns
    df["ds"] = df["timestamp_utc"].dt.tz_convert(None)  # remove tzinfo but keep UTC hour
    df["y"] = df["intensity"].astype(float)

    cols = [
        "ds",
        "y",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "is_holiday",
        "dow",
        "hour",
    ]
    df = df[cols].dropna()

    print(f"Training dataset for {counter_id}: {len(df)} rows")
    return df


def train_prophet_with_regressors(df_train: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model with regressors on df_train.
    """
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )

    regressors: List[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "is_holiday",
        "dow",
        "hour",
    ]

    for reg in regressors:
        m.add_regressor(reg)

    m.fit(df_train)
    return m


def build_future_dataframe(
    df_forecast_weather: pd.DataFrame,
    df_holidays: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Prophet 'future' dataframe from weather forecast (UTC).
      - ds
      - same regressors as training
    """
    df = df_forecast_weather.copy()
    df["ds"] = df["timestamp_utc"].dt.tz_convert(None)


    df["date"] = df["timestamp_utc"].dt.date
    df = df.merge(
        df_holidays[["date", "name"]],
        on="date",
        how="left",
        suffixes=("", "_holiday"),
    )
    df["is_holiday"] = df["name"].notna().astype(int)

    df["dow"] = df["timestamp_utc"].dt.weekday
    df["hour"] = df["timestamp_utc"].dt.hour

    df_future = df[
        [
            "ds",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "is_holiday",
            "dow",
            "hour",
        ]
    ].copy()

    return df_future


# ---------- FULL PIPELINE FOR ALL COUNTERS ----------


def predict_for_all_counters():
    """
    Full pipeline:
      - load shared data (weather history, holidays, forecast)
      - loop on SELECTED_COUNTERS
        - load bike history
        - build training df
        - train Prophet
        - build future df
        - predict tomorrow
      - upsert all predictions into bike_predictions_hourly
    """
    print("🔄 Loading shared data from Supabase...")
    df_weather = load_weather_history()
    df_holidays = load_holidays()
    df_forecast_weather = load_weather_forecast_for_tomorrow()

    all_preds = []

    for counter_id in SELECTED_COUNTERS:
        print("\n🚴 Processing counter:", counter_id)

        try:
            df_bike = load_bike_history(counter_id)
        except ValueError as e:
            print(f"  ⚠️ Skipping {counter_id}: {e}")
            continue

        df_train = build_training_dataframe(
            counter_id, df_bike, df_weather, df_holidays
        )

        if df_train.empty:
            print(f"  ⚠️ Empty training set for {counter_id}, skipping.")
            continue

        model = train_prophet_with_regressors(df_train)
        df_future = build_future_dataframe(df_forecast_weather, df_holidays)
        forecast = model.predict(df_future)

        df_pred = pd.DataFrame({
            "counter_id": counter_id,
            "timestamp_utc": df_future["ds"],
            "yhat": forecast["yhat"],
            "yhat_lower": forecast["yhat_lower"],
            "yhat_upper": forecast["yhat_upper"],
        })

        all_preds.append(df_pred)

    if not all_preds:
        print("❌ No predictions produced for any counter.")
        return

    df_all_preds = pd.concat(all_preds, ignore_index=True)

    # Convert timestamps to ISO string for Supabase JSON
    df_all_preds["timestamp_utc"] = df_all_preds["timestamp_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    print("\n📊 Predictions preview:")
    print(df_all_preds.head())

    res = upsert_df("bike_predictions_hourly_prophet", df_all_preds)
    print("\n✅ Upsert into bike_predictions_hourly_prophet:", res)


def main():
    predict_for_all_counters()


if __name__ == "__main__":
    main()
