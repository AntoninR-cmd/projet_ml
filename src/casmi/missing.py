"""Valeurs manquantes : nettoyage, règles de reconstruction et outils de masquage.

Les NA des métadonnées dépendent presque entièrement de la bibliothèque
(voir docs/dictionnaire_donnees.md). Ce module fournit :
- le nettoyage des faux NA et le regroupement des instruments en familles ;
- le résumé de collision_energy_ev (une liste par spectre) en colonnes numériques ;
- la règle exacte NCE → eV, qui reconstruit une partie des NA ;
- deux façons de masquer des valeurs observées pour évaluer un imputeur :
  au hasard, ou par bibliothèque entière (le vrai mécanisme des NA).
"""

import re

import numpy as np
import pandas as pd

# Textes utilisés à la place d'une vraie valeur manquante
FAKE_NA = ["nan", "N/A", ""]

# Familles d'instruments, testées dans l'ordre : la première qui correspond gagne
INSTRUMENT_FAMILIES = [
    ("timsTOF", r"tims"),
    ("Orbitrap/FT", r"orbitrap|exactive|qehf|ft\b|qft|itft|hybrid ft"),
    ("QTOF/TOF", r"tof|maxis|impact|synapt"),
    ("Trappe ionique", r"ion trap|iontrap|\blit\b|qit|ltq|-it\b"),
    ("Quadripôle", r"qqq|qq\b|quattro|-q\b"),
]


def clean_fake_na(s):
    """Remplace les faux NA ("nan", "N/A", chaîne vide) par de vrais NA."""
    return s.mask(s.isin(FAKE_NA))


def instrument_family(s):
    """Regroupe les libellés bruts d'instrument en quelques familles.

    Les faux NA deviennent NA ; un libellé connu mais d'aucune famille devient "Autre".
    """
    s = clean_fake_na(s)
    lower = s.str.lower()
    family = pd.Series(pd.NA, index=s.index, dtype="object")
    for name, pattern in INSTRUMENT_FAMILIES:
        hit = family.isna() & lower.str.contains(pattern, regex=True, na=False)
        family[hit] = name
    family[family.isna() & s.notna()] = "Autre"
    return family.astype("category")


def ce_summary(ce_ev):
    """Résume la liste collision_energy_ev en colonnes : n_ce, ce_min, ce_max, ce_mean.

    Un spectre sans énergie (NA) donne NaN partout, et n_ce = 0.
    """
    n = ce_ev.map(lambda v: 0 if v is None or (np.isscalar(v) and pd.isna(v)) else len(v))
    stats = {"ce_min": np.min, "ce_max": np.max, "ce_mean": np.mean}
    out = pd.DataFrame({"n_ce": n.astype("int64")}, index=ce_ev.index)
    for name, f in stats.items():
        out[name] = [f(v) if k > 0 else np.nan for v, k in zip(ce_ev, n)]
    return out


def nce_to_ev(orig, units, precursor_mz):
    """Convertit une énergie NCE simple (ex. "35" ou "35%") en eV.

    Règle vérifiée sur tous les spectres du train qui ont les deux valeurs :
    ev = NCE × precursor_mz / 500, arrondi au dixième. Renvoie NaN quand la
    règle ne s'applique pas (autre unité, plage "30->50%", liste…).
    """
    text = orig.astype("string").str.strip().str.rstrip("%")
    nce = pd.to_numeric(text.where(text.str.fullmatch(r"\d+(\.\d+)?", na=False)), errors="coerce")
    nce = nce.where(units == "NCE")
    return (nce * precursor_mz / 500).round(1).astype("float64")


def mask_random(observed, frac, seed):
    """Masque au hasard une fraction des valeurs observées.

    observed : Series booléenne (True = valeur présente). Renvoie un masque booléen.
    """
    rng = np.random.default_rng(seed)
    return observed & (rng.random(len(observed)) < frac)


def library_folds(observed, libraries, min_observed=1_000):
    """Un masque par bibliothèque : toutes ses valeurs observées sont masquées.

    On simule ainsi une bibliothèque qui ne renseignerait pas la variable, ce qui
    est le vrai mécanisme des NA. Seules les bibliothèques avec au moins
    min_observed valeurs observées forment un pli. Renvoie {bibliothèque: masque}.
    """
    counts = libraries[observed].value_counts()
    return {lib: observed & (libraries == lib) for lib in counts[counts >= min_observed].index}
