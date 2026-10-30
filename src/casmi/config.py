"""Constantes du projet : chemins et paramètres partagés.

Ce module ne fait que définir des valeurs : l'importer ne télécharge rien,
ne crée aucun dossier et n'affiche rien.
"""

from pathlib import Path

# Racine du projet, calculée depuis ce fichier (indépendante du dossier courant)
ROOT = Path(__file__).resolve().parents[2]  # config.py → casmi → src → projet_ml

COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"

# --- Données ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

TRAIN_PATH = DATA_RAW / "train.parquet"
TEST_PATH = DATA_RAW / "test.parquet"
SUBMISSION_PATH = DATA_RAW / "sample_submission.csv"

# --- Reproductibilité ---
SEED = 42
