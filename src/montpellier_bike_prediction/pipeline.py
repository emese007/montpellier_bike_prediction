# pipeline.py

from __future__ import annotations

from datetime import date

from montpellier_bike_prediction.db_supabase import truncate_table

# --- ETL : APIs -> CSV (historique) ---
from montpellier_bike_prediction.etl.bike_etl import main as bike_etl_main
from montpellier_bike_prediction.etl.weather_etl import main as weather_etl_main
from montpellier_bike_prediction.etl.holidays_etl import main as holidays_etl_main

# --- CLI : CSV -> Supabase ---
from montpellier_bike_prediction.cli.load_counters import run as load_counters
from montpellier_bike_prediction.cli.load_weather_hourly import (
    main as load_weather_hourly,
)
from montpellier_bike_prediction.cli.load_bike_hourly import main as load_bike_hourly
from montpellier_bike_prediction.cli.load_holidays import main as load_holidays

# --- Forecast météo (API -> Supabase directement) ---
from montpellier_bike_prediction.cli.update_weather_forecast import (
    main as update_weather_forecast,
)

# --- Modèles ---
from montpellier_bike_prediction.cli.train_and_predict_prophet import (
    main as run_prophet_pipeline,
)
from montpellier_bike_prediction.cli.train_and_predict_xgboost import (
    main as run_xgb_pipeline,
)


# -------------------------------------------------------------------
# 1) Règle métier : quand régénérer les jours fériés ?
# -------------------------------------------------------------------


def should_refresh_holidays() -> bool:
    """
    Retourne True si on doit régénérer les jours fériés.

    Règle :
    - si on est le 30 ou 31 décembre : on régénère pour inclure l'année suivante.
    """
    today = date.today()
    return today.month == 12 and today.day >= 30


# -------------------------------------------------------------------
# 2) ETL : APIs -> CSV (historique)
# -------------------------------------------------------------------


def run_etl_history():
    """
    Lance les ETL pour recréer les fichiers CSV d'entrée à partir des APIs.
    (historique uniquement : vélo, météo, jours fériés)
    """
    print("\nETL BIKE: API -> CSV")
    bike_etl_main()

    print("\nETL WEATHER HISTORY: API -> CSV")
    weather_etl_main()

    if should_refresh_holidays():
        print("\nETL HOLIDAYS: API -> CSV (yearly refresh)")
        holidays_etl_main()
    else:
        print("\nETL HOLIDAYS: skipped (not end of December)")


# -------------------------------------------------------------------
# 3) CSV -> Supabase (historique)
# -------------------------------------------------------------------


def reload_data_to_supabase():
    """
    Vide les tables historiques et recharge les données depuis les CSV.
    """
    print("\nTruncating historical tables in Supabase...")

    # Tables historiques vélo & météo & forecast météo & compteurs
    truncate_table("bike_hourly", "timestamp_utc", "1900-01-01T00:00:00+00")
    truncate_table("weather_hourly", "timestamp_utc", "1900-01-01T00:00:00+00")

    truncate_table("counters", "id", "")
    truncate_table("weather_forecast_hourly", "timestamp_utc", "1900-01-01T00:00:00+00")

    # Jours fériés: on ne les vide QUE si on les a regénérés cette fois-ci
    if should_refresh_holidays():
        truncate_table("holidays", "date", "1900-01-01T00:00:00+00")
        reload_holidays = True
    else:
        reload_holidays = False

    print("\nLoading history from CSV to Supabase...")

    # Ces loaders lisent data/processed/*.csv
    load_counters()
    load_weather_hourly()
    load_bike_hourly()
    update_weather_forecast()

    if reload_holidays:
        load_holidays()
    else:
        print("Holidays: skipped reload (already up to date)")


# -------------------------------------------------------------------
# 4) Lancer les modèles
# -------------------------------------------------------------------


def run_models():
    """
    Lance Prophet et XGBoost pour tous les compteurs, à partir des données
    déjà présentes dans Supabase (bike_hourly, weather_hourly, holidays,
    weather_forecast_hourly).
    """
    # On vide les anciennes prédictions pour n'avoir que celles de demain
    truncate_table(
        "bike_predictions_hourly_prophet", "timestamp_utc", "1900-01-01T00:00:00+00"
    )
    truncate_table(
        "bike_predictions_hourly_xgboost", "timestamp_utc", "1900-01-01T00:00:00+00"
    )

    print("\nRunning Prophet pipeline (all counters)...")
    run_prophet_pipeline()

    print("\nRunning XGBoost pipeline (all counters)...")
    run_xgb_pipeline()


# -------------------------------------------------------------------
# 6) Pipeline globale
# -------------------------------------------------------------------


def main():
    print("Starting FULL Montpellier Bike Prediction pipeline")

    # 1) APIs historiques -> CSV
    run_etl_history()

    # 2) CSV -> Supabase (historique)
    reload_data_to_supabase()

    # 3) Modèles de prédiction
    run_models()

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
