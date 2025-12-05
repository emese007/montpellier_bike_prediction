# pipeline.py
"""
Pipeline principale pour le projet Montpellier Bike Prediction.

Usage (depuis la racine du projet) :

    # Revenir à un état propre (optionnel) + tout recharger et prédire
    uv run python pipeline.py --reset-all --reload-history

    # En usage quotidien : juste mettre à jour la prévision météo et les prédictions
    uv run python pipeline.py
"""

from __future__ import annotations

import argparse

from montpellier_bike_prediction.db_supabase import get_supabase_client

# --- CLI des différentes étapes ---
from montpellier_bike_prediction.cli.load_counters import run as load_counters
from montpellier_bike_prediction.cli.load_holidays import main as load_holidays
from montpellier_bike_prediction.cli.load_weather_hourly import (
    main as load_weather_hourly,
)
from montpellier_bike_prediction.cli.load_bike_hourly import main as load_bike_hourly
from montpellier_bike_prediction.cli.update_weather_forecast import (
    main as update_weather_forecast,
)
from montpellier_bike_prediction.cli.train_and_predict_prophet import (
    main as run_prophet_pipeline,
)
from montpellier_bike_prediction.cli.train_and_predict_xgboost import (
    main as run_xgb_pipeline,
)


# -------------------------------------------------------------------
# 1) Helpers pour nettoyer les tables Supabase (optionnels)
# -------------------------------------------------------------------


def _delete_all_rows(table: str, any_column: str) -> None:
    """
    Supprime toutes les lignes d'une table en utilisant un filtre large.
    On utilise `col != '__CLEAR__'` qui est toujours vrai.
    """
    client = get_supabase_client()
    print(f"🧹 Clearing table {table} ...")
    (
        client.table(table)
        .delete()
        .neq(any_column, "__CLEAR__")  # condition toujours vraie
        .execute()
    )


def reset_predictions_tables() -> None:
    """
    Vide uniquement les tables de prédictions + forecast météo.
    Utile à lancer tous les jours si tu veux que ces tables ne contiennent
    que les prédictions du prochain jour.
    """
    _delete_all_rows("weather_forecast_hourly", "timestamp_utc")
    _delete_all_rows("bike_predictions_hourly_prophet", "counter_id")
    _delete_all_rows("bike_predictions_hourly_xgboost", "counter_id")


def reset_all_tables() -> None:
    """
    Vide TOUTES les tables du projet (sauf éventuellement des trucs système).
    À utiliser seulement quand tu veux tout reconstruire depuis les CSV.
    """
    # Tables "de base"
    _delete_all_rows("bike_hourly", "counter_id")
    _delete_all_rows("weather_hourly", "timestamp_utc")
    _delete_all_rows("holidays", "date")
    _delete_all_rows("counters", "id")

    # Tables de prédictions / forecast
    reset_predictions_tables()


# -------------------------------------------------------------------
# 2) Étapes logiques de la pipeline
# -------------------------------------------------------------------


def load_static_history_from_csv() -> None:
    """
    Recharge les données historiques à partir des CSV dans /data/processed :

      - bike_selected_hourly_processed.csv  -> bike_hourly
      - weather_hourly_processed.csv        -> weather_hourly
      - holidays_processed.csv              -> holidays
      - compteurs                           -> counters (via API 3M)

    À appeler surtout quand tu as regénéré les CSV depuis tes notebooks/ETL.
    """
    print("\n📥 (1) Loading counters from Montpellier API → Supabase")
    load_counters()

    print("\n📥 (2) Loading holidays CSV → Supabase")
    load_holidays()

    print("\n📥 (3) Loading weather history CSV → Supabase")
    load_weather_hourly()

    print("\n📥 (4) Loading bike history CSV → Supabase")
    load_bike_hourly()


def update_forecast_and_predictions() -> None:
    """
    Étape quotidienne typique :

      1) Met à jour la prévision météo de demain (UTC) dans weather_forecast_hourly
      2) Entraîne Prophet sur l'historique + prévoit demain
      3) Entraîne XGBoost sur l'historique + prévoit demain
    """
    print("\n🌦 (5) Updating weather forecast for tomorrow (UTC) → Supabase")
    update_weather_forecast()

    print("\n📈 (6) Training + predicting with Prophet → bike_predictions_hourly_prophet")
    run_prophet_pipeline()

    print("\n🤖 (7) Training + predicting with XGBoost → bike_predictions_hourly_xgboost")
    run_xgb_pipeline()


# -------------------------------------------------------------------
# 3) Point d'entrée principal
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Montpellier Bike Prediction pipeline")
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Vide toutes les tables du projet avant de recharger (historique + prédictions).",
    )
    parser.add_argument(
        "--reset-predictions",
        action="store_true",
        help="Vide seulement les tables de prédictions + forecast météo.",
    )
    parser.add_argument(
        "--reload-history",
        action="store_true",
        help="Recharge l'historique depuis les CSV (bike, météo, jours fériés, compteurs).",
    )

    args = parser.parse_args()

    # 0) Reset éventuel
    if args.reset_all:
        print("\nRESET COMPLET DES TABLES SUPABASE")
        reset_all_tables()
    elif args.reset_predictions:
        print("\nRESET DES TABLES DE PREDICTIONS")
        reset_predictions_tables()

    # 1) Recharger l'historique si demandé
    if args.reload_history:
        load_static_history_from_csv()

    # 2) Toujours : mettre à jour la prévision + faire les prédictions
    update_forecast_and_predictions()

    print("\nPipeline terminée.")


if __name__ == "__main__":
    main()
