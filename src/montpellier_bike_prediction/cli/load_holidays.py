from pathlib import Path
import pandas as pd
from montpellier_bike_prediction.db_supabase import upsert_df

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "holidays_processed.csv"
)


def load_holidays_csv() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = df[["date", "name", "year"]]  # juste les colonnes utiles

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["year"] = df["year"].astype(int)

    return df


def main():
    print(f"➡ Loading CSV from: {DATA_PATH}")
    df_holidays = load_holidays_csv()
    print(df_holidays.head())

    res = upsert_df("holidays", df_holidays)
    print("✔ Upload terminé")
    print(res)


if __name__ == "__main__":
    main()
