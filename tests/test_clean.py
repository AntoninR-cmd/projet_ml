import numpy as np
import pandas as pd
import pytest

from casmi.clean import (
    META_PATH, STRUCTURE_RULES, filters, load_clean_meta, rule_adduct_unknown, rule_charge_vs_mode,
    rule_dubious_adduct, rule_mass_error,
)
from casmi.config import TRAIN_PATH

GLUCOSE = 180.06339  # masse exacte de C6H12O6
PROTON = 1.007276


def test_adduct_inconnu():
    df = pd.DataFrame({"adduct": ["[Cat]2+", "[M+H]+"], "precursor_mz": [200.0, 200.0]})
    assert rule_adduct_unknown(df).tolist() == [True, False]


def test_charge_contre_mode():
    df = pd.DataFrame({
        "adduct": ["[M-H]-", "[M+H]+", "[M+H]+", "[M-H]-", "[Cat]2+"],
        "ionization_mode": ["positive", "positive", "negative", "negative", "positive"],
    })
    # Un adduct non interprétable n'est pas compté ici (règle adduct_inconnu)
    assert rule_charge_vs_mode(df).tolist() == [True, False, True, False, False]


def test_erreur_masse():
    df = pd.DataFrame({
        "adduct": ["[M+H]+", "[M+H]+", "[M+H]+", "[Cat]2+"],
        "precursor_mz": [
            GLUCOSE + PROTON,           # étiquette juste
            round(GLUCOSE + PROTON),    # précurseur arrondi à l'unité (basse résolution) : toléré
            GLUCOSE + 22.989218,        # en réalité [M+Na]+ : erreur de 21,98 Da
            200.0,                      # adduct inconnu : compté par une autre règle
        ],
    })
    exact = pd.Series(GLUCOSE, index=df.index)
    assert rule_mass_error(df, exact).tolist() == [False, False, True, False]


def test_adduct_douteux():
    df = pd.DataFrame({"adduct": ["[M-CH3]-", "[M-H]-"]})
    assert rule_dubious_adduct(df).tolist() == [True, False]


def test_filters_sans_structure_pour_les_requetes():
    rules = pd.DataFrame({
        "adduct_inconnu": [False, True, False],
        "charge_contre_mode": [False, False, False],
        "erreur_masse": [False, False, True],
        "adduct_douteux": [False, False, False],
    })
    out = filters(rules)
    assert out["keep"].tolist() == [True, False, False]
    # Une règle qui utilise la vraie structure ne doit pas filtrer les requêtes de validation
    assert STRUCTURE_RULES == ["erreur_masse"]
    assert out["query_ok"].tolist() == [True, False, True]


@pytest.mark.skipif(not (TRAIN_PATH.exists() and META_PATH.exists()), reason="données ou meta.parquet absents")
def test_meta_parquet():
    """Toutes les lignes du train, row_id intact, keep plus strict que query_ok."""
    import pyarrow.parquet as pq

    meta = load_clean_meta()
    assert len(meta) == pq.ParquetFile(TRAIN_PATH).metadata.num_rows
    assert np.array_equal(meta["row_id"].to_numpy(), np.arange(len(meta)))
    assert not (meta["keep"] & ~meta["query_ok"]).any()
