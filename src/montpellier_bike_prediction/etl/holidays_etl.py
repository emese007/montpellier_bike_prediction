# src/montpellier_bike_traffic/etl/holidays_etl.py

from __future__ import annotations

from pathlib import Path

from .holidays_client import HolidaysAPIClient


DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_holidays_etl(start_year: int = 2023, zone: str = "metropole") -> None:
    """
    ETL jours fériés : récupère les jours fériés depuis start_year
    jusqu'à l'année courante et sauvegarde un CSV.
    """
    client = HolidaysAPIClient()
    df = client.fetch_range_for_training(start_year=start_year, zone=zone)

    raw_path = DATA_RAW_DIR / "holidays_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Jours fériés bruts sauvegardés dans {raw_path}")

    # Peu de nettoyage à faire, on recopie dans processed
    processed_path = DATA_PROCESSED_DIR / "holidays_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"✅ Jours fériés traités sauvegardés dans {processed_path}")


def main():
    """Entry point used by the pipeline and when run as a script."""
    run_holidays_etl()

if __name__ == "__main__":
    main()
    
