import numpy as np
import pandas as pd
import pytest

from casmi.catalog import smiles_properties

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
