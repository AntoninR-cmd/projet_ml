import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from casmi.matrices import load_matrices, save_subset, select_rows


def make_meta():
    """10 spectres : 0-5 dans le split train, 6-9 dans des molécules dev."""
    return pd.DataFrame({
        "row_id": np.arange(10),
        "inchikey14": list("AABBCCDDEE"),
        "split": ["train"] * 6 + ["val"] * 4,
        "dev": [False] * 6 + [True] * 4,
        "keep": [True, True, True, False, True, True, True, False, True, True],
        "query_ok": [True] * 7 + [False, True, True],
    })


def test_select_rows():
    rows = select_rows(make_meta(), n_train_sample=3, seed=0)
    assert rows["dev"].tolist() == [6, 8, 9]  # query_ok, pas keep (7 est écarté par query_ok)
    assert len(rows["train_sample"]) == 3
    assert set(rows["train_sample"]) <= {0, 1, 2, 4, 5}  # split train et keep
    assert rows["train_sample"].tolist() == sorted(rows["train_sample"])
    # Même graine, même tirage
    assert rows["train_sample"].tolist() == select_rows(make_meta(), 3, seed=0)["train_sample"].tolist()


def test_echantillon_plus_grand_que_le_train():
    assert len(select_rows(make_meta(), n_train_sample=100)["train_sample"]) == 5


@pytest.fixture
def saved(tmp_path):
    rows = make_meta().iloc[[6, 8, 9]]
    X = sp.csr_matrix(np.array([[0.6, 0.8, 0], [0, 0, 1.0], [1.0, 0, 0]], dtype=np.float32))
    metas = {
        "onehot": pd.DataFrame({"mode_positif": [1.0, 0.0, 1.0], "adduct=[M+H]+": [1.0, 0.0, 1.0]}, index=rows.index),
        "decompose": pd.DataFrame({"mode_positif": [1.0, 0.0, 1.0], "adduct_nM": [1.0, 2.0, 1.0]}, index=rows.index),
    }
    save_subset("dev", rows, X, metas, directory=tmp_path)
    return tmp_path


def test_aller_retour(saved):
    X, rows, names = load_matrices("dev", directory=saved)
    assert rows["row_id"].tolist() == [6, 8, 9]
    assert rows["inchikey14"].tolist() == ["D", "E", "E"]
    assert names == ["bin_0", "bin_1", "bin_2", "mode_positif", "adduct=[M+H]+"]
    assert X.toarray()[1].tolist() == [0, 0, 1, 0, 0]


def test_encodage_et_spectres_seuls(saved):
    _, _, names = load_matrices("dev", adduct_encoding="decompose", directory=saved)
    assert names[-1] == "adduct_nM"
    X, _, names = load_matrices("dev", meta=False, directory=saved)
    assert X.shape == (3, 3) and names == ["bin_0", "bin_1", "bin_2"]


def test_desalignement_detecte(saved):
    meta = pd.read_parquet(saved / "meta_dev_onehot.parquet")
    meta.iloc[::-1].to_parquet(saved / "meta_dev_onehot.parquet", index=False)
    with pytest.raises(ValueError):
        load_matrices("dev", directory=saved)


def test_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_matrices("dev", directory=tmp_path)
