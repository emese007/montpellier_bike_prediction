from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from montpellier_bike_prediction.db_supabase import upsert_df

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_DIR / "data" / "processed" / "bike_selected_hourly_processed.csv"


def load_bike_csv() -> pd.DataFrame:
    """
    Charge le CSV des séries vélo sélectionnées et adapte les colonnes
    au schéma de la table bike_hourly :

      - counter_id (FK vers counters.id)
      - timestamp_utc (timestamptz)
      - intensity (int)
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"📁 Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Colonnes vélo:", df.columns.tolist())

    # Timestamp
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    elif "timestamp" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise ValueError(
            "Colonne 'timestamp_utc' ou 'timestamp' manquante dans le CSV vélo."
        )

    # Identifiant compteur
    if "counter_id" in df.columns:
        # déjà bon
        pass
    elif "ecocounter_id" in df.columns:
        df["counter_id"] = df["ecocounter_id"]
    elif "eco_id" in df.columns:
        df["counter_id"] = df["eco_id"]
    else:
        raise ValueError(
            "Colonne 'counter_id', 'ecocounter_id' ou 'eco_id' manquante dans le CSV vélo."
        )

    if "intensity" not in df.columns:
        raise ValueError("Colonne 'intensity' manquante dans le CSV vélo.")

    # Types simples
    df["intensity"] = df["intensity"].astype(int)

    df_bike = df[["counter_id", "timestamp_utc", "intensity"]].copy()

    # Unicité
    df_bike = df_bike.drop_duplicates(subset=["counter_id", "timestamp_utc"])

    # Conversion en string ISO pour JSON
    df_bike["timestamp_utc"] = df_bike["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return df_bike


def upsert_bike_hourly(df_bike: pd.DataFrame, chunk_size: int = 2000):
    total = len(df_bike)
    if total == 0:
        print("Aucune ligne vélo à envoyer.")
        return

    print(f"🚴 Envoi de {total} lignes vers bike_hourly (paquets de {chunk_size})")

    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk = df_bike.iloc[start:end].copy()

        res = upsert_df("bike_hourly", chunk)
        print(f"  Chunk {i + 1}/{n_chunks} : {res.get('count')} lignes upsertées")


def main():
    print(f"➡ Loading bike CSV from: {DATA_PATH}")
    df_bike = load_bike_csv()
    print(df_bike.head())
    upsert_bike_hourly(df_bike)


if __name__ == "__main__":
    main()
