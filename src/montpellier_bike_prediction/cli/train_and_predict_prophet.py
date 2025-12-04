from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd
from prophet import Prophet

from montpellier_bike_prediction.db_supabase import get_supabase_client, upsert_df


# Tu peux changer ce compteur par défaut
DEFAULT_COUNTER_ID = "urn:ngsi-ld:EcoCounter:X2H22104775"


# ---------- Chargement des données depuis Supabase ----------


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
        raise ValueError(f"Aucune donnée dans bike_hourly pour {counter_id}")

    df = pd.DataFrame(rows)
    # Conversion en datetime UTC
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def load_weather_history() -> pd.DataFrame:
    """
    Charge toute la météo horaire historique.
    Table : weather_hourly(timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m)
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
        raise ValueError("Aucune donnée dans weather_hourly")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def load_holidays() -> pd.DataFrame:
    """
    Charge le calendrier des jours fériés.
    Table : holidays(date, name, year)
    """
    client = get_supabase_client()
    resp = client.table("holidays").select("date, name, year").execute()

    rows = resp.data or []
    if not rows:
        raise ValueError("Aucune donnée dans holidays")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_weather_forecast() -> pd.DataFrame:
    """
    Charge les prévisions horaires de la table weather_forecast_hourly
    pour demain (UTC).
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
        raise ValueError("Aucune donnée dans weather_forecast_hourly")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


# ---------- Construction du dataset d'entraînement ----------


def build_training_dataframe(counter_id: str) -> pd.DataFrame:
    """
    Construit le DataFrame d'entraînement pour Prophet avec régressseurs :
      - ds : timestamp_utc (UTC)
      - y  : intensity
      - temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m
      - is_holiday, dow (jour de la semaine), hour (heure)
    """
    df_bike = load_bike_history(counter_id)
    df_weather = load_weather_history()
    df_holidays = load_holidays()

    # Jointure vélo + météo sur timestamp_utc
    df = pd.merge(
        df_bike,
        df_weather,
        on="timestamp_utc",
        how="left",
    )

    # Création de la date UTC pour jointure avec jours fériés
    df["date"] = df["timestamp_utc"].dt.date
    df = df.merge(
        df_holidays[["date", "name"]],
        on="date",
        how="left",
        suffixes=("", "_holiday"),
    )

    df["is_holiday"] = df["name"].notna().astype(int)

    # Features temporelles
    df["dow"] = df["timestamp_utc"].dt.weekday  # 0=lundi, 6=dimanche
    df["hour"] = df["timestamp_utc"].dt.hour

    # Renommer pour Prophet
    df["ds"] = df["timestamp_utc"]
    df["y"] = df["intensity"].astype(float)

    # Garder seulement les colonnes utiles
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

    print(f"Dataset d'entraînement pour {counter_id} : {len(df)} lignes")
    return df


# ---------- Entraînement et prédiction Prophet ----------


def train_prophet_with_regressors(df_train: pd.DataFrame) -> Prophet:
    """
    Entraîne un modèle Prophet sur df_train avec régressseurs explicites.
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


def build_future_dataframe(df_forecast_weather: pd.DataFrame, df_holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le DataFrame 'future' pour Prophet à partir des prévisions météo UTC
    pour demain (24h).

      - ds = timestamp_utc (UTC)
      - mêmes régressseurs que pour l'entraînement
    """
    df = df_forecast_weather.copy()
    df["ds"] = df["timestamp_utc"]

    # Date UTC pour jours fériés
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

    # Garder les colonnes nécessaires
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


def predict_tomorrow_for_counter(counter_id: str):
    """
    Pipeline complet :
      1) construit df_train pour un compteur
      2) entraîne Prophet
      3) charge les prévisions météo de demain
      4) construit future df pour Prophet
      5) prédit yhat, yhat_lower, yhat_upper
      6) upsert dans bike_predictions_hourly
    """
    print(f"🚴 Training + predicting for counter: {counter_id}")

    df_train = build_training_dataframe(counter_id)
    if df_train.empty:
        raise ValueError("Dataset d'entraînement vide")

    model = train_prophet_with_regressors(df_train)

    df_forecast_weather = load_weather_forecast()
    df_holidays = load_holidays()

    # Optionnel : s'assurer qu'on ne garde que 'demain' en UTC
    now_utc = datetime.now(timezone.utc)
    tomorrow = (now_utc + timedelta(days=1)).date()
    df_forecast_weather["date"] = df_forecast_weather["timestamp_utc"].dt.date
    df_forecast_weather = df_forecast_weather[df_forecast_weather["date"] == tomorrow].copy()

    if df_forecast_weather.empty:
        raise ValueError("Pas de météo de prévision pour demain dans weather_forecast_hourly")

    df_future = build_future_dataframe(df_forecast_weather, df_holidays)

    forecast = model.predict(df_future)

    df_pred = pd.DataFrame({
        "counter_id": counter_id,
        "timestamp_utc": df_future["ds"],
        "yhat": forecast["yhat"],
        "yhat_lower": forecast["yhat_lower"],
        "yhat_upper": forecast["yhat_upper"],
    })

    # Conversion en string ISO pour JSON -> Supabase
    df_pred["timestamp_utc"] = df_pred["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    print("Prévisions (extrait) :")
    print(df_pred.head())

    res = upsert_df("bike_predictions_hourly", df_pred)
    print("Upsert bike_predictions_hourly:", res)


def main():
    # pour l'instant on ne gère qu'un compteur (tu pourras boucler sur 10)
    predict_tomorrow_for_counter(DEFAULT_COUNTER_ID)


if __name__ == "__main__":
    main()
