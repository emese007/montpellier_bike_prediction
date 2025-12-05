# src/montpellier_bike_traffic/db_supabase.py

from __future__ import annotations

from typing import Any

import pandas as pd
from supabase import create_client, Client

from .config import SUPABASE_URL, SUPABASE_KEY


def get_supabase_client() -> Client:
    """
    Initialise et renvoie un client Supabase à partir des variables d'environnement.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL ou SUPABASE_KEY manquants. "
            "Définis-les dans tes variables d'environnement."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def upsert_df(table: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    Upsert un DataFrame dans une table Supabase.
    Les colonnes du DataFrame doivent correspondre aux colonnes SQL.
    """
    if df.empty:
        return {"status": "empty", "table": table}

    client = get_supabase_client()
    records = df.to_dict(orient="records")

    resp = client.table(table).upsert(records).execute()
    return {
        "status": "ok",
        "table": table,
        "count": len(records),
        "response": resp.data,
    }


def truncate_table(table: str) -> dict[str, Any]:
    """
    Vide complètement une table Supabase via l'API REST.
    Équivaut à TRUNCATE TABLE, mais en utilisant DELETE sans condition.

    Nécessite que les politiques RLS autorisent le delete pour ton rôle.
    """
    client = get_supabase_client()
    print(f"Truncating table '{table}'...")

    try:
        resp = client.table(table).delete().execute()
    except Exception as e:
        print(f"Failed to truncate {table}: {e}")
        return {"status": "error", "table": table, "error": str(e)}

    return {
        "status": "ok",
        "table": table,
        # selon la version, resp.count peut être None, ce n'est pas grave
        "deleted_count": getattr(resp, "count", "unknown"),
    }
