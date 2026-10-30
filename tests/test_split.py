import numpy as np
import pandas as pd
import pytest

from casmi.config import TRAIN_PATH
from casmi.split import SPLIT_PATH, load_split, make_split


def fake_inchikeys(n_molecules=5_000, seed=0):
    """Des inchikey14 factices, avec plusieurs spectres par molécule, dans le désordre."""
    rng = np.random.default_rng(seed)
    keys = np.array([f"KEY{i:011d}" for i in range(n_molecules)], dtype=object)
    return rng.permutation(np.repeat(keys, rng.integers(1, 6, n_molecules)))


def test_une_ligne_par_molecule():
    inchikeys = fake_inchikeys()
    split = make_split(inchikeys, n_dev=100)
    assert split["inchikey14"].is_unique
    assert set(split["inchikey14"]) == set(inchikeys)


def test_tailles():
    split = make_split(fake_inchikeys(), n_dev=100)
    assert (split["split"] == "val").sum() == 500  # 10 % de 5 000
    assert split["dev"].sum() == 100
    assert set(split["split"]) == {"train", "val"}


def test_dev_inclus_dans_val():
    split = make_split(fake_inchikeys(), n_dev=100)
    assert (split.loc[split["dev"], "split"] == "val").all()


def test_reproductible():
    inchikeys = fake_inchikeys()
    pd.testing.assert_frame_equal(make_split(inchikeys, n_dev=100), make_split(inchikeys, n_dev=100))


def test_independant_de_l_ordre_des_lignes():
    inchikeys = fake_inchikeys()
    shuffled = np.random.default_rng(1).permutation(inchikeys)
    pd.testing.assert_frame_equal(make_split(inchikeys, n_dev=100), make_split(shuffled, n_dev=100))


def test_la_graine_change_le_tirage():
    inchikeys = fake_inchikeys()
    a = make_split(inchikeys, n_dev=100, seed=1)
    b = make_split(inchikeys, n_dev=100, seed=2)
    assert not a["split"].equals(b["split"])


def test_trop_de_dev():
    with pytest.raises(ValueError):
        make_split(fake_inchikeys(n_molecules=100), n_dev=50)


@pytest.mark.skipif(not (TRAIN_PATH.exists() and SPLIT_PATH.exists()), reason="données ou split.parquet absents")
def test_fichier_fige():
    """split.parquet couvre tout le train et se régénère à l'identique."""
    from casmi.io import load_meta

    meta = load_meta(["inchikey14"])
    expected = make_split(meta["inchikey14"])
    pd.testing.assert_frame_equal(load_split(), expected)

    # Aucune molécule des deux côtés : chaque spectre reçoit exactement un split
    spectra = meta.merge(load_split(), on="inchikey14", how="left", validate="many_to_one")
    assert spectra["split"].notna().all()
