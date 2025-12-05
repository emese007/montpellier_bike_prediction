from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from montpellier_bike_prediction.config import SELECTED_COUNTERS
from montpellier_bike_prediction.db_supabase import get_supabase_client, upsert_df


# ---------- LOADERS FROM SUPABASE ----------


def load_bike_history(counter_id: str) -> pd.DataFrame:
    """
    Charge l'historique vélo horaire pour un compteur donné.
    Table : bike_hourly(counter_id, timestamp_utc, intensity)
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
    df["intensity"] = df["intensity"].astype(float)
    return df


def load_weather_history() -> pd.DataFrame:
    """
    Charge toute la météo horaire historique.
    Table : weather_hourly(timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m)
    """
    client = get_supabase_client()
    resp = (
        client.table("weather_hourly")
        .select(
            "timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m"
        )
        .order("timestamp_utc")
        .execute()
    )

    rows = resp.data or []
    if not rows:
        raise ValueError("No data in weather_hourly")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # casts sécurité
    for col in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
        df[col] = df[col].astype(float)

    return df


def load_holidays() -> pd.DataFrame:
    """
    Charge les jours fériés.
    Table : holidays(date, name, year)
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
    Charge les prévisions horaires météo de weather_forecast_hourly
    et garde uniquement demain (UTC).
    """
    client = get_supabase_client()
    resp = (
        client.table("weather_forecast_hourly")
        .select(
            "timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m"
        )
        .order("timestamp_utc")
        .execute()
    )

    rows = resp.data or []
    if not rows:
        raise ValueError("No data in weather_forecast_hourly")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    for col in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
        df[col] = df[col].astype(float)

    now_utc = datetime.now(timezone.utc)
    tomorrow = (now_utc + timedelta(days=1)).date()
    df["date"] = df["timestamp_utc"].dt.date
    df = df[df["date"] == tomorrow].copy()

    if df.empty:
        raise ValueError("No forecast rows for tomorrow in weather_forecast_hourly")

    # Tri et reset index
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


# ---------- FEATURE ENGINEERING ----------


FEATURE_COLS = [
    "hour",
    "dow",
    "is_holiday",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]


def build_training_dataset(
    counter_id: str,
    df_bike: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_holidays: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit le dataset d'entraînement pour XGBoost pour un compteur :
      - target: intensity
      - features:
          - hour, dow (jour de semaine)
          - is_holiday (0/1)
          - météo: temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m
    Tout reste en UTC.
    """
    # Join vélo + météo
    df = pd.merge(
        df_bike,
        df_weather,
        on="timestamp_utc",
        how="left",
    )

    # Jours fériés par date (UTC)
    df["date"] = df["timestamp_utc"].dt.date
    df = df.merge(
        df_holidays[["date", "name"]],
        on="date",
        how="left",
        suffixes=("", "_holiday"),
    )

    df["is_holiday"] = df["name"].notna().astype(int)
    df["hour"] = df["timestamp_utc"].dt.hour
    df["dow"] = df["timestamp_utc"].dt.weekday  # 0=lundi, 6=dimanche

    # On garde uniquement des lignes complètes
    df = df.dropna(subset=["intensity"] + FEATURE_COLS)

    print(f"[XGB] Training dataset for {counter_id}: {len(df)} rows")
    return df


def build_future_features(
    df_forecast_weather: pd.DataFrame,
    df_holidays: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit le dataframe de features pour la prédiction de demain (24h UTC) :
      - timestamp_utc
      - mêmes features que pour l'entraînement (hour, dow, is_holiday + météo)
    """
    df = df_forecast_weather.copy()
    df["date"] = df["timestamp_utc"].dt.date

    df = df.merge(
        df_holidays[["date", "name"]],
        on="date",
        how="left",
        suffixes=("", "_holiday"),
    )
    df["is_holiday"] = df["name"].notna().astype(int)
    df["hour"] = df["timestamp_utc"].dt.hour
    df["dow"] = df["timestamp_utc"].dt.weekday

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


# ---------- TRAIN + PREDICT XGBOOST ----------


def train_xgb_regressor(df_train: pd.DataFrame) -> XGBRegressor:
    """
    Entraîne un XGBRegressor simple sur df_train.
    """
    X = df_train[FEATURE_COLS].values
    y = df_train["intensity"].values

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(X, y)
    return model


def predict_for_all_counters_xgb():
    """
    Pipeline complet XGBoost :

      1) Charge :
         - météo historique
         - jours fériés
         - prévision météo de demain (UTC)
      2) Pour chaque compteur de SELECTED_COUNTERS :
         - charge bike_hourly
         - construit le dataset d'entraînement
         - entraîne un XGBRegressor
         - construit les features pour demain
         - prédit l'intensité horaire (24h)
      3) Concatène toutes les prédictions et upsert dans
         bike_predictions_hourly_xgboost.
    """
    print("🔄 [XGB] Loading shared data from Supabase...")
    df_weather = load_weather_history()
    df_holidays = load_holidays()
    df_forecast_weather = load_weather_forecast_for_tomorrow()

    all_preds = []

    for counter_id in SELECTED_COUNTERS:
        print(f"\n🚴 [XGB] Processing counter: {counter_id}")

        try:
            df_bike = load_bike_history(counter_id)
        except ValueError as e:
            print(f"  ⚠️ Skipping {counter_id}: {e}")
            continue

        df_train = build_training_dataset(counter_id, df_bike, df_weather, df_holidays)

        if len(df_train) < 100:
            print(f"  ⚠️ Not enough data for {counter_id} (rows={len(df_train)}), skipping.")
            continue

        model = train_xgb_regressor(df_train)

        df_future = build_future_features(df_forecast_weather, df_holidays)
        X_future = df_future[FEATURE_COLS].values

        y_pred = model.predict(X_future)

        df_pred = pd.DataFrame({
            "counter_id": counter_id,
            "timestamp_utc": df_future["timestamp_utc"],
            "yhat": y_pred,
        })

        all_preds.append(df_pred)

    if not all_preds:
        print("❌ [XGB] No predictions produced for any counter.")
        return

    df_all_preds = pd.concat(all_preds, ignore_index=True)

    # Conversion en string ISO pour JSON -> Supabase
    df_all_preds["timestamp_utc"] = df_all_preds["timestamp_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    print("\n📊 [XGB] Predictions preview:")
    print(df_all_preds.head())

    # Upsert dans la table dédiée XGBoost
    res = upsert_df("bike_predictions_hourly_xgboost", df_all_preds)
    print("\n✅ [XGB] Upsert into bike_predictions_hourly_xgboost:", res)


def main():
    predict_for_all_counters_xgb()


if __name__ == "__main__":
    main()
