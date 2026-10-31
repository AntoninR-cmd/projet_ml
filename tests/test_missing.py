import numpy as np
import pandas as pd

from casmi.missing import ce_summary, clean_fake_na, instrument_family, library_folds, mask_random, nce_to_ev


def test_clean_fake_na():
    s = pd.Series(["timsTOF", "nan", "N/A", "", None])
    assert clean_fake_na(s).isna().tolist() == [False, True, True, True, True]


def test_instrument_family():
    s = pd.Series(["timsTOF", "Orbitrap", "LC-ESI-QFT", "qTof", "LC-ESI-ITTOF", "Linear Ion Trap", "LC-ESI-QQ", "FAB-EBEB", "N/A"])
    expected = ["timsTOF", "Orbitrap/FT", "Orbitrap/FT", "QTOF/TOF", "QTOF/TOF", "Trappe ionique", "Quadripôle", "Autre", None]
    result = instrument_family(s).astype("object").where(lambda x: x.notna(), None).tolist()
    assert result == expected


def test_ce_summary():
    ce = pd.Series([[20.0, 40.0, 60.0], [35.0], None])
    out = ce_summary(ce)
    assert out["n_ce"].tolist() == [3, 1, 0]
    assert out.loc[0, ["ce_min", "ce_max", "ce_mean"]].tolist() == [20.0, 60.0, 40.0]
    assert out.loc[2, ["ce_min", "ce_max", "ce_mean"]].isna().all()


def test_nce_to_ev():
    orig = pd.Series(["30", "35%", "30->50%", "40", None])
    units = pd.Series(["NCE", "NCE", "NCE", "eV", "NCE"])
    mz = pd.Series([242.1434, 500.0, 500.0, 500.0, 500.0])
    ev = nce_to_ev(orig, units, mz)
    assert ev[0] == 14.5  # exemple réel du train
    assert ev[1] == 35.0
    assert ev[2:].isna().all()  # plage, autre unité, NA


def test_mask_random():
    observed = np.array([True] * 900 + [False] * 100)
    mask = mask_random(observed, 0.2, seed=0)
    assert not mask[900:].any()  # on ne masque que des valeurs observées
    assert 140 < mask.sum() < 220


def test_library_folds():
    libs = pd.Series(["a"] * 2000 + ["b"] * 2000 + ["c"] * 10)
    observed = pd.Series([True] * 2000 + [False] * 2000 + [True] * 10)
    folds = library_folds(observed, libs)
    assert list(folds) == ["a"]  # b n'a rien d'observé, c a trop peu de valeurs
    assert folds["a"].sum() == 2000
