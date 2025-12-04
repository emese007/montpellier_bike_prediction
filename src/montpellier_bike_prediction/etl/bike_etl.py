# src/montpellier_bike_traffic/etl/bike_etl.py

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from ..config import SELECTED_COUNTERS
from .bike_client import BikeAPIClient


DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_year_chunks(start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime, dt.datetime]]:
    """
    Construit des (start, end) annuels entre deux datetimes.
    Évite de faire un appel géant sur plusieurs années.
    """
    chunks: list[tuple[dt.datetime, dt.datetime]] = []
    current = start
    while current < end:
        next_year = dt.datetime(current.year + 1, 1, 1, tzinfo=current.tzinfo)
        chunk_end = min(next_year - dt.timedelta(seconds=1), end)
        chunks.append((current, chunk_end))
        current = next_year
    return chunks


def fetch_bike_history_for_counters(
    counters: Iterable[str],
    global_start: str = "2023-01-01T00:00:00",
    global_end: str | None = None,
) -> pd.DataFrame:
    """
    Récupère l'historique des intensités pour une liste de compteurs
    entre global_start et global_end (UTC, ISO).

    Renvoie un DataFrame (timestamp_utc, intensity, ecocounter_id).
    """
    client = BikeAPIClient()

    start_dt = dt.datetime.fromisoformat(global_start).replace(tzinfo=dt.timezone.utc)
    if global_end is None:
        now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc)
        end_dt = now
    else:
        end_dt = dt.datetime.fromisoformat(global_end).replace(tzinfo=dt.timezone.utc)

    chunks = build_year_chunks(start_dt, end_dt)
    all_dfs: list[pd.DataFrame] = []

    for cid in tqdm(list(counters), desc="Counters"):
        for (s, e) in chunks:
            df_ts = client.fetch_timeseries(cid, s, e)
            if df_ts is not None and not df_ts.empty:
                all_dfs.append(df_ts)

    if not all_dfs:
        return pd.DataFrame(columns=["timestamp_utc", "intensity", "ecocounter_id"])

    df = pd.concat(all_dfs, ignore_index=True)
    df = (
        df
        .drop_duplicates(subset=["ecocounter_id", "timestamp_utc"])
        .sort_values(["ecocounter_id", "timestamp_utc"])
        .reset_index(drop=True)
    )
    return df


def run_bike_etl_for_selected(
    global_start: str = "2023-01-01T00:00:00",
    global_end: str | None = None,
) -> None:
    """
    ETL complet vélo pour les 10 compteurs sélectionnés.
    Sauvegarde le résultat dans data/raw/ et data/processed/
    """
    print("🚴 ETL vélo - compteurs sélectionnés")
    df = fetch_bike_history_for_counters(SELECTED_COUNTERS, global_start, global_end)

    raw_path = DATA_RAW_DIR / "bike_selected_hourly_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Données vélo brutes sauvegardées dans {raw_path}")

    # Ici tu pourras ajouter des traitements (fusion d'IDs, etc.)
    processed_path = DATA_PROCESSED_DIR / "bike_selected_hourly_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"✅ Données vélo traitées sauvegardées dans {processed_path}")


if __name__ == "__main__":
    run_bike_etl_for_selected()
