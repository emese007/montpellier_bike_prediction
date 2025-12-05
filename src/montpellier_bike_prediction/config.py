# src/montpellier_bike_prediction/config.py

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root automatically
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# À ADAPTER avec la vraie liste de tes 10 compteurs choisis
SELECTED_COUNTERS = [
    "urn:ngsi-ld:EcoCounter:X2H22043034",
    "urn:ngsi-ld:EcoCounter:X2H22043035",
    "urn:ngsi-ld:EcoCounter:X2H22104768",
    "urn:ngsi-ld:EcoCounter:X2H22104774",
    "urn:ngsi-ld:EcoCounter:X2H22104775",
    "urn:ngsi-ld:EcoCounter:X2H22104776",
    "urn:ngsi-ld:EcoCounter:X2H22104773",
    "urn:ngsi-ld:EcoCounter:X2H20042635",
    "urn:ngsi-ld:EcoCounter:X2H22104769",
    "urn:ngsi-ld:EcoCounter:X2H22104766",
]

# Coordonnées centrales de Montpellier (pour la météo par ex.)
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", 43.6))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", 3.88))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
