"""Split figé train / validation, par molécule (inchikey14).

Tous les spectres d'une molécule sont du même côté : une molécule de validation
n'a aucun spectre dans l'entraînement, comme les molécules du test.

Pour (re)générer data/processed/split.parquet :
    python -m casmi.split
"""

import numpy as np
import pandas as pd

from casmi.config import DATA_PROCESSED, SEED
from casmi.io import load_meta

SPLIT_PATH = DATA_PROCESSED / "split.parquet"

VAL_FRACTION = 0.10  # part des molécules en validation
N_DEV = 2_000  # molécules de validation du sous-ensemble dev (itérations rapides)


def make_split(inchikeys, val_fraction=VAL_FRACTION, n_dev=N_DEV, seed=SEED):
    """Une ligne par inchikey14, avec split ("train" ou "val") et dev (booléen).

    inchikeys : les inchikey14 des spectres (les doublons sont permis).
    Le résultat ne dépend que de l'ensemble des molécules et de la graine :
    les clés sont triées avant le tirage, donc l'ordre des lignes en entrée
    ne change rien.
    """
    keys = np.unique(np.asarray(inchikeys, dtype=object))  # uniques et triées
    n_val = int(val_fraction * len(keys))
    if n_dev > n_val:
        raise ValueError(f"n_dev ({n_dev}) dépasse le nombre de molécules de validation ({n_val})")

    rng = np.random.default_rng(seed)
    val_keys = rng.choice(keys, size=n_val, replace=False)
    dev_keys = rng.choice(np.sort(val_keys), size=n_dev, replace=False)

    split = pd.DataFrame({"inchikey14": keys})
    split["split"] = np.where(split["inchikey14"].isin(val_keys), "val", "train")
    split["dev"] = split["inchikey14"].isin(dev_keys)
    return split


def build_split(path=SPLIT_PATH):
    """Calcule le split sur tout le train et l'enregistre en parquet."""
    meta = load_meta(["inchikey14"])
    split = make_split(meta["inchikey14"])

    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(path, index=False)
    return split


def load_split(path=SPLIT_PATH):
    """Charge le split figé : inchikey14, split, dev."""
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancez `python -m casmi.split`.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    split = build_split()
    counts = split.groupby("split").size()
    print(f"Split enregistré dans {SPLIT_PATH}")
    print(f"  train : {counts['train']:,} molécules")
    print(f"  val   : {counts['val']:,} molécules, dont {split['dev'].sum():,} en dev")
