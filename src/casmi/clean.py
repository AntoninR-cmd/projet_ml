"""Nettoyage du train : meta.parquet et sa colonne keep.

Une ligne par spectre du train (tous les row_id sont conservés) avec deux
colonnes de filtre :
- keep : le spectre est fiable pour l'entraînement (toutes les règles) ;
- query_ok : le spectre peut servir de requête de validation. Seules les règles
  qui n'utilisent que des informations présentes dans le test comptent ici :
  filtrer la validation avec la vraie structure la rendrait plus facile que le test.

Pour (re)générer data/processed/meta.parquet :
    python -m casmi.clean
"""

import numpy as np
import pandas as pd

from casmi.catalog import smiles_properties
from casmi.config import DATA_PROCESSED
from casmi.io import load_meta
from casmi.mass import compute_neutral_masses, parse_adduct

META_PATH = DATA_PROCESSED / "meta.parquet"

# Colonnes du contrat d'interfaces, puis les deux filtres
META_COLUMNS = [
    "row_id", "inchikey14", "normalized_smiles", "ingest_lib", "ionization_mode", "instrument_type",
    "adduct", "precursor_mz", "collision_energy_ev", "num_peaks",
]

# Une erreur d'adduct décale la masse d'au moins un atome d'hydrogène (1,008 Da) ;
# un precursor_mz arrondi à l'unité (appareils à basse résolution) s'écarte d'au plus 0,5 Da.
MAX_MASS_ERROR_DA = 0.75

# Adducts dont les précurseurs ne correspondent à aucune formule (conclusions de B1.1)
DUBIOUS_ADDUCTS = ["[M-CH3]-"]

# Règles qui utilisent la vraie structure : autorisées pour nettoyer l'entraînement,
# interdites pour filtrer les requêtes de validation
STRUCTURE_RULES = ["erreur_masse"]


def rule_adduct_unknown(df):
    """L'adduct n'est pas interprétable : pas de masse neutre (ex. "[Cat]2+")."""
    return pd.Series(np.isnan(compute_neutral_masses(df)), index=df.index)


def _charge_sign(adduct):
    try:
        return np.sign(parse_adduct(adduct)[2])
    except ValueError:
        return 0


def rule_charge_vs_mode(df):
    """La charge de l'adduct contredit le mode d'ionisation (ex. "[M-H]-" en mode positif).

    C'est physiquement impossible : l'un des deux est mal étiqueté.
    """
    signs = {a: _charge_sign(a) for a in df["adduct"].dropna().unique()}
    sign = df["adduct"].map(signs)
    expected = df["ionization_mode"].map({"positive": 1, "negative": -1})
    return (sign != 0) & expected.notna() & (sign != expected)


def mass_error_da(df, exact_mass):
    """Écart (Da) entre la masse neutre déduite du précurseur et la masse exacte de la structure."""
    return pd.Series(compute_neutral_masses(df), index=df.index) - exact_mass


def rule_mass_error(df, exact_mass, max_error=MAX_MASS_ERROR_DA):
    """La masse mesurée ne correspond pas à la structure annoncée (étiquette fausse).

    exact_mass : masse exacte du SMILES de chaque spectre (Series alignée sur df).
    Un adduct non interprétable (masse NaN) n'est pas compté ici, mais par rule_adduct_unknown.
    """
    return mass_error_da(df, exact_mass).abs() > max_error


def rule_dubious_adduct(df, adducts=DUBIOUS_ADDUCTS):
    """Adduct dont les précurseurs sont peu fiables dans les données."""
    return df["adduct"].isin(adducts)


def exact_masses(smiles):
    """Masse exacte (RDKit) de chaque SMILES, chaque SMILES distinct n'étant calculé qu'une fois."""
    props = smiles_properties(smiles.unique())
    return smiles.map(props["exact_mass"])


def cleaning_rules(df, exact_mass):
    """Un booléen par règle (True = spectre à écarter), dans l'ordre d'application."""
    return pd.DataFrame({
        "adduct_inconnu": rule_adduct_unknown(df),
        "charge_contre_mode": rule_charge_vs_mode(df),
        "erreur_masse": rule_mass_error(df, exact_mass),
        "adduct_douteux": rule_dubious_adduct(df),
    })


def filters(rules):
    """keep (aucune règle déclenchée) et query_ok (aucune règle sans structure déclenchée)."""
    return pd.DataFrame({
        "keep": ~rules.any(axis=1),
        "query_ok": ~rules.drop(columns=STRUCTURE_RULES).any(axis=1),
    })


def build_meta(path=META_PATH):
    """Construit meta.parquet : métadonnées du train, keep et query_ok."""
    meta = load_meta(META_COLUMNS[1:])
    rules = cleaning_rules(meta, exact_masses(meta["normalized_smiles"]))
    meta = pd.concat([meta[META_COLUMNS], filters(rules)], axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(path, index=False)
    return meta


def load_clean_meta(path=META_PATH):
    """Charge meta.parquet (une ligne par spectre du train, avec keep et query_ok)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancez `python -m casmi.clean`.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    meta = build_meta()
    print(f"meta.parquet enregistré dans {META_PATH}")
    print(f"  spectres : {len(meta):,}")
    print(f"  keep     : {meta['keep'].sum():,} ({meta['keep'].mean():.2%})")
    print(f"  query_ok : {meta['query_ok'].sum():,} ({meta['query_ok'].mean():.2%})")
