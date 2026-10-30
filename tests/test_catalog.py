import numpy as np
import pandas as pd
import pytest

from casmi.catalog import (
    build_catalog,
    choose_smiles,
    get_candidates,
    load_catalog,
    mass_windows,
    smiles_properties,
    smiles_to_inchikey14,
)

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
ACETIC_ACID = "CC(=O)O"


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_smiles_properties(n_jobs):
    props = smiles_properties([CAFFEINE, "pas un smiles", ACETIC_ACID], n_jobs=n_jobs, chunksize=1)

    assert list(props.index) == [CAFFEINE, "pas un smiles", ACETIC_ACID]
    assert props.loc[CAFFEINE, "formula"] == "C8H10N4O2"
    # C8H10N4O2 : 8 × 12 + 10 × 1,00782503 + 4 × 14,00307401 + 2 × 15,99491462
    assert props.loc[CAFFEINE, "exact_mass"] == pytest.approx(194.080376, abs=1e-6)
    assert props.loc[ACETIC_ACID, "formula"] == "C2H4O2"
    assert props.loc[ACETIC_ACID, "exact_mass"] == pytest.approx(60.021129, abs=1e-6)
    assert pd.isna(props.loc["pas un smiles", "formula"])
    assert np.isnan(props.loc["pas un smiles", "exact_mass"])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_smiles_to_inchikey14(n_jobs):
    keys = smiles_to_inchikey14([CAFFEINE, "pas un smiles", ACETIC_ACID], n_jobs=n_jobs)
    # InChIKey de la caféine : RYYVLZVUVIJVGH-UHFFFAOYSA-N
    assert keys[CAFFEINE] == "RYYVLZVUVIJVGH"
    assert keys[ACETIC_ACID] == "QTBSBXVTEAMEQO"
    assert pd.isna(keys["pas un smiles"])


def test_choose_smiles_most_frequent_then_alphabetical():
    pairs = pd.DataFrame({
        "inchikey14": ["K1", "K1", "K1", "K2", "K2"],
        "normalized_smiles": ["B", "A", "B", "Z", "Y"],
    })
    chosen = choose_smiles(pairs)
    assert chosen["K1"] == "B"  # 2 spectres contre 1
    assert chosen["K2"] == "Y"  # égalité : ordre alphabétique


@pytest.fixture
def small_catalog(tmp_path):
    # Un mini « train » : 3 molécules, dont une avec deux SMILES, plus un SMILES illisible
    train = pd.DataFrame({
        "inchikey14": ["CAFE", "CAFE", "ACET", "ACET", "ACET", "BAD"],
        "normalized_smiles": [CAFFEINE, CAFFEINE, ACETIC_ACID, ACETIC_ACID, "OC(C)=O", "pas un smiles"],
        "precursor_mz": [195.0, 195.1, 61.0, 61.0, 61.0, 100.0],
    })
    train.to_parquet(tmp_path / "train.parquet", index=False)
    out = tmp_path / "processed" / "catalog.parquet"
    return build_catalog(tmp_path / "train.parquet", out, n_jobs=1), out


def test_build_catalog(small_catalog):
    catalog, out = small_catalog

    assert list(catalog.columns) == ["cat_idx", "inchikey14", "smiles", "formula", "exact_mass"]
    # Trié par masse : acide acétique (60) puis caféine (194) ; le SMILES illisible est écarté
    assert list(catalog["inchikey14"]) == ["ACET", "CAFE"]
    assert list(catalog["cat_idx"]) == [0, 1]
    assert catalog.loc[0, "smiles"] == ACETIC_ACID  # le plus fréquent des deux SMILES
    assert catalog.loc[1, "formula"] == "C8H10N4O2"

    loaded = load_catalog(out)
    pd.testing.assert_frame_equal(loaded, catalog)


def test_mass_windows_bounds():
    masses = np.array([100.0, 199.999, 200.0, 200.002, 200.0021, 300.0])
    # 10 ppm de 200 Da = ± 0,002 Da : de 199,998 à 200,002 inclus
    lo, hi = mass_windows([200.0, 50.0, np.nan], masses, ppm=10)
    assert (lo[0], hi[0]) == (1, 4)
    assert lo[1] == hi[1]  # aucune masse aussi petite
    assert lo[2] == hi[2]  # masse inconnue : aucun candidat


def test_get_candidates():
    masses = np.array([100.0, 199.999, 200.0, 200.002, 200.0021, 300.0])

    np.testing.assert_array_equal(get_candidates(200.0, ppm=10, masses=masses), [1, 2, 3])
    np.testing.assert_array_equal(get_candidates(200.0, ppm=1, masses=masses), [2])
    np.testing.assert_array_equal(get_candidates(200.0, ppm=20, masses=masses), [1, 2, 3, 4])
    assert get_candidates(np.nan, masses=masses).size == 0


def test_get_candidates_from_catalog_file(small_catalog):
    catalog, out = small_catalog
    masses = load_catalog(out)["exact_mass"].to_numpy()
    caffeine_mass = catalog.loc[1, "exact_mass"]

    np.testing.assert_array_equal(get_candidates(caffeine_mass, ppm=5, masses=masses), [1])
