"""Masse neutre d'une molécule à partir du précurseur et de l'adduct (étape 1 du pipeline).

Un adduct s'écrit [nM ± X ± Y]z± : n molécules M, des groupes ajoutés (+) ou perdus (-),
puis la charge z. La masse de l'ion vaut |z| × m/z = n × M + delta − z × m_e, d'où :

    M = (|z| × m/z − delta + z × m_e) / n

Le parser est volontairement strict : un adduct qu'il ne comprend pas entièrement lève
une ValueError, plutôt que de renvoyer une masse fausse.
"""

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem

ELECTRON_MASS = 0.000548579909065  # Da (CODATA 2018)

_PERIODIC_TABLE = Chem.GetPeriodicTable()

# Un nombre sans zéro en tête et jamais nul : "2", "11", mais pas "0" ni "02"
_COUNT = r"[1-9]\d*"
# Une formule brute : uniquement des blocs Élément + nombre ("C4H11O2", "Na")
_FORMULA = rf"(?:[A-Z][a-z]?(?:{_COUNT})?)+"
# Un terme ajouté ou perdu, avec un coefficient éventuel : "+H", "-2H2O"
_TERM = rf"([+-])({_COUNT})?({_FORMULA})"

_ADDUCT_RE = re.compile(rf"\[({_COUNT})?M((?:[+-](?:{_COUNT})?{_FORMULA})*)\]({_COUNT})?([+-])")
_TERM_RE = re.compile(_TERM)
_FORMULA_RE = re.compile(_FORMULA)
_ELEMENT_RE = re.compile(rf"([A-Z][a-z]?)({_COUNT})?")


@lru_cache(maxsize=None)
def element_mass(symbol: str) -> float:
    """Masse monoisotopique d'un élément (isotope le plus abondant), en Da."""
    try:
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(symbol)
    except RuntimeError:
        atomic_number = 0

    # RDKit accepte aussi "D" et "T" comme symboles de l'hydrogène : on vérifie que le
    # symbole est bien celui de l'élément, sinon le deutérium aurait la masse de H.
    if atomic_number <= 0 or _PERIODIC_TABLE.GetElementSymbol(atomic_number) != symbol:
        raise ValueError(f"Élément inconnu : {symbol!r}")

    return _PERIODIC_TABLE.GetMostCommonIsotopeMass(atomic_number)


def formula_mass(formula: str) -> float:
    """Masse monoisotopique d'une formule brute : "H2O" -> 2 × m(H) + m(O)."""
    if _FORMULA_RE.fullmatch(formula) is None:
        raise ValueError(f"Formule non interprétable : {formula!r}")

    return sum(
        element_mass(symbol) * int(count or 1)
        for symbol, count in _ELEMENT_RE.findall(formula)
    )


def parse_adduct(adduct: str) -> tuple[int, float, int]:
    """Décompose un adduct en (n_mol, delta, z).

    - n_mol : nombre de molécules M dans l'ion ("[2M+Na]+" -> 2) ;
    - delta : somme signée des masses ajoutées (+) et perdues (-), en Da ;
    - z : charge signée ("]+" -> 1, "]2-" -> -2).

    Exemple : parse_adduct("[M-2H2O+H]+") -> (1, -2 × m(H2O) + m(H), 1).

    Lève ValueError si l'adduct n'est pas entièrement interprétable (ex. "[Cat]2+").
    """
    if not isinstance(adduct, str):
        raise ValueError(f"Adduct non interprétable : {adduct!r}")

    match = _ADDUCT_RE.fullmatch(adduct)
    if match is None:
        raise ValueError(f"Adduct non interprétable : {adduct!r}")

    n_mol, terms, charge, sign = match.groups()

    delta = 0.0
    for term_sign, coef, formula in _TERM_RE.findall(terms):
        mass = int(coef or 1) * formula_mass(formula)
        delta += mass if term_sign == "+" else -mass

    z = int(charge or 1) * (1 if sign == "+" else -1)

    return int(n_mol or 1), delta, z


def _try_parse_adduct(adduct: str) -> tuple[float, float, float]:
    try:
        return parse_adduct(adduct)
    except ValueError:
        return (np.nan, np.nan, np.nan)


def compute_neutral_masses(df: pd.DataFrame) -> np.ndarray:
    """Masse neutre M (en Da) de chaque spectre, à partir des colonnes precursor_mz et adduct.

    Chaque adduct distinct n'est parsé qu'une seule fois, puis la formule est appliquée
    à tout le tableau. Renvoie NaN pour les spectres dont l'adduct est manquant ou non
    interprétable.
    """
    params = pd.DataFrame.from_dict(
        {adduct: _try_parse_adduct(adduct) for adduct in df["adduct"].dropna().unique()},
        orient="index",
        columns=["n_mol", "delta", "z"],
    )
    # Une ligne de paramètres par spectre, dans l'ordre de df (NaN si adduct inconnu)
    p = params.reindex(df["adduct"].to_numpy())

    n_mol = p["n_mol"].to_numpy(dtype=float)
    delta = p["delta"].to_numpy(dtype=float)
    z = p["z"].to_numpy(dtype=float)
    precursor_mz = df["precursor_mz"].to_numpy(dtype=float)

    return (np.abs(z) * precursor_mz - delta + z * ELECTRON_MASS) / n_mol
