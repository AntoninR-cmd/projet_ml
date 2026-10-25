"""Catalogue de structures connues et filtre par masse (étape 2 du pipeline).

Le catalogue contient une ligne par molécule (inchikey14), triée par masse exacte
croissante : cat_idx = 0, 1, 2… suit donc l'ordre des masses, et les candidats d'une
fenêtre de masse forment un intervalle contigu de cat_idx.

Usage en ligne de commande, pour (re)construire catalog.parquet :
    python -m casmi.catalog
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# Par défaut, un processus par cœur logique de la machine
N_JOBS = os.cpu_count() or 1

# Chemins provisoires, en attendant config.py (A) : les données brutes sont dans
# CASMI_DATA_DIR (ou kaggle/input), les fichiers produits dans son sous-dossier
# processed (ou data/processed).
_ROOT = Path(__file__).resolve().parents[2]
if "CASMI_DATA_DIR" in os.environ:
    RAW_DIR = Path(os.environ["CASMI_DATA_DIR"])
    PROCESSED_DIR = RAW_DIR / "processed"
else:
    RAW_DIR = _ROOT / "kaggle" / "input"
    PROCESSED_DIR = _ROOT / "data" / "processed"
CATALOG_PATH = PROCESSED_DIR / "catalog.parquet"

CATALOG_COLUMNS = ["cat_idx", "inchikey14", "smiles", "formula", "exact_mass"]


def _properties_chunk(smiles: list[str]) -> list[tuple[str | None, float]]:
    # Exécutée dans chaque processus : RDKit signale les SMILES invalides sur stderr
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append((None, np.nan))
        else:
            out.append((rdMolDescriptors.CalcMolFormula(mol), Descriptors.ExactMolWt(mol)))
    return out


def smiles_properties(smiles, n_jobs: int = N_JOBS, chunksize: int = 5_000) -> pd.DataFrame:
    """Formule brute et masse exacte (monoisotopique, en Da) de chaque SMILES.

    Le calcul est réparti entre n_jobs processus. Renvoie un DataFrame indexé par
    les SMILES, avec les colonnes formula et exact_mass (NaN si RDKit ne sait pas
    lire le SMILES).
    """
    smiles = list(smiles)
    chunks = [smiles[i : i + chunksize] for i in range(0, len(smiles), chunksize)]

    if n_jobs == 1:
        results = [r for chunk in chunks for r in _properties_chunk(chunk)]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = [r for part in executor.map(_properties_chunk, chunks) for r in part]

    return pd.DataFrame(
        results, columns=["formula", "exact_mass"], index=pd.Index(smiles, name="smiles")
    )


def choose_smiles(pairs: pd.DataFrame) -> pd.Series:
    """Choisit un SMILES par inchikey14 : le plus fréquent parmi les spectres.

    pairs contient une ligne par spectre (colonnes inchikey14, normalized_smiles).
    En cas d'égalité, on prend le premier SMILES dans l'ordre alphabétique, pour
    que le choix soit reproductible.
    """
    counts = pairs.groupby(["inchikey14", "normalized_smiles"], observed=True).size().rename("n").reset_index()
    counts = counts.sort_values(["inchikey14", "n", "normalized_smiles"], ascending=[True, False, True])
    return counts.drop_duplicates("inchikey14").set_index("inchikey14")["normalized_smiles"]


def build_catalog(
    train_path: Path | None = None, out_path: Path | None = None, n_jobs: int = N_JOBS
) -> pd.DataFrame:
    """Construit le catalogue à partir du train et l'enregistre en parquet.

    Une ligne par inchikey14, avec un SMILES choisi par choose_smiles, et la formule
    et la masse exacte calculées par RDKit depuis ce SMILES (et non depuis le
    précurseur, qui contient l'erreur de mesure). Les molécules dont le SMILES est
    illisible sont écartées. Le catalogue est trié par masse croissante.
    """
    train_path = Path(train_path or RAW_DIR / "train.parquet")
    out_path = Path(out_path or CATALOG_PATH)

    pairs = pq.read_table(train_path, columns=["inchikey14", "normalized_smiles"]).to_pandas()
    smiles = choose_smiles(pairs)
    props = smiles_properties(smiles.to_numpy(), n_jobs=n_jobs)

    catalog = pd.DataFrame({
        "inchikey14": smiles.index.to_numpy(),
        "smiles": smiles.to_numpy(),
        "formula": props["formula"].to_numpy(),
        "exact_mass": props["exact_mass"].to_numpy(),
    })
    catalog = catalog.dropna(subset=["exact_mass"])
    catalog = catalog.sort_values(["exact_mass", "inchikey14"], ignore_index=True)
    catalog.insert(0, "cat_idx", np.arange(len(catalog)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_path, index=False)
    _load_catalog_cached.cache_clear()
    return catalog


@lru_cache(maxsize=None)
def _load_catalog_cached(path: str) -> pd.DataFrame:
    catalog = pd.read_parquet(path)
    if not catalog["exact_mass"].is_monotonic_increasing:
        raise ValueError(f"Catalogue non trié par masse : {path}")
    return catalog


def load_catalog(path: Path | None = None) -> pd.DataFrame:
    """Charge le catalogue (cat_idx, inchikey14, smiles, formula, exact_mass).

    Le fichier n'est lu qu'une fois par session : les appels suivants renvoient
    le même DataFrame, qu'il ne faut donc pas modifier.
    """
    return _load_catalog_cached(str(Path(path or CATALOG_PATH)))


def mass_windows(neutral_masses, masses: np.ndarray, ppm: float = 10) -> tuple[np.ndarray, np.ndarray]:
    """Bornes [lo, hi) des candidats de chaque masse, dans un tableau de masses trié.

    Les candidats de neutral_masses[i] sont masses[lo[i]:hi[i]], c'est-à-dire les
    masses comprises dans [M − δ, M + δ] avec δ = M × ppm × 10⁻⁶. Une masse NaN
    donne une fenêtre vide.
    """
    neutral_masses = np.asarray(neutral_masses, dtype=float)
    delta = neutral_masses * ppm * 1e-6
    lo = np.searchsorted(masses, neutral_masses - delta, side="left")
    hi = np.searchsorted(masses, neutral_masses + delta, side="right")
    hi = np.where(np.isnan(neutral_masses), lo, hi)
    return lo, hi


def get_candidates(neutral_mass: float, ppm: float = 10, masses: np.ndarray | None = None) -> np.ndarray:
    """cat_idx des molécules du catalogue dont la masse est à ± ppm de neutral_mass.

    Par défaut, cherche dans le catalogue chargé par load_catalog(). masses permet
    de fournir un autre tableau de masses trié, dont la position i vaut cat_idx = i.
    """
    if masses is None:
        masses = load_catalog()["exact_mass"].to_numpy()
    lo, hi = mass_windows([neutral_mass], masses, ppm)
    return np.arange(lo[0], hi[0])


if __name__ == "__main__":
    t0 = time.perf_counter()
    catalog = build_catalog()
    print(f"{len(catalog):,} molécules écrites dans {CATALOG_PATH} en {time.perf_counter() - t0:.0f} s")
