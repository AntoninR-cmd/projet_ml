"""Catalogue de structures connues et filtre par masse (étape 2 du pipeline)."""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# Par défaut, un processus par cœur logique de la machine
N_JOBS = os.cpu_count() or 1


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
