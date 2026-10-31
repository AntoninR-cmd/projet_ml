"""Matrices prêtes à l'emploi pour les premiers modèles de B et C.

Deux sous-ensembles :
- dev : les spectres des 2 000 molécules dev, filtrés avec query_ok (requêtes de validation) ;
- train_sample : 200 000 spectres tirés au hasard dans le split train, filtrés avec keep.

Pour chacun, dans data/processed/ :
- X_{nom}.npz : les spectres binés (matrice creuse, 10 000 colonnes, lignes de norme 1) ;
- meta_{nom}_{encodage}.parquet : les features de métadonnées, pour chaque encodage de l'adduct ;
- rows_{nom}.parquet : row_id et inchikey14 de chaque ligne, dans le même ordre.
Les encodeurs sont appris sur tout le split train (keep), puis sauvegardés
(meta_encoder_{encodage}.pkl) pour encoder plus tard la validation et le test.

Pour (re)générer les fichiers :
    python -m casmi.matrices
Pour les lire : load_matrices("dev") ou load_matrices("train_sample").
"""

import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

from casmi.clean import load_clean_meta
from casmi.config import DATA_PROCESSED, SEED
from casmi.features import ADDUCT_ENCODINGS, META_INPUT_COLUMNS, bin_spectra, fit_meta_encoder, meta_features
from casmi.io import load_meta, load_spectra
from casmi.split import load_split

N_TRAIN_SAMPLE = 200_000


def select_rows(meta, n_train_sample=N_TRAIN_SAMPLE, seed=SEED):
    """row_id (triés) de chaque sous-ensemble.

    meta : une ligne par spectre, avec row_id, split, dev, keep et query_ok.
    """
    train = meta.loc[(meta["split"] == "train") & meta["keep"], "row_id"].to_numpy()
    rng = np.random.default_rng(seed)
    return {
        "dev": np.sort(meta.loc[meta["dev"] & meta["query_ok"], "row_id"].to_numpy()),
        "train_sample": np.sort(rng.choice(train, min(n_train_sample, len(train)), replace=False)),
    }


def save_subset(name, rows, X, metas, directory=DATA_PROCESSED):
    """Sauvegarde un sous-ensemble.

    rows : DataFrame avec row_id et inchikey14 ; X : spectres binés ;
    metas : {encodage: features de métadonnées}. Tout est aligné ligne à ligne sur rows.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rows = rows[["row_id", "inchikey14"]].reset_index(drop=True)
    sp.save_npz(directory / f"X_{name}.npz", sp.csr_matrix(X))
    rows.to_parquet(directory / f"rows_{name}.parquet", index=False)
    for encoding, features in metas.items():
        features = features.reset_index(drop=True)
        features.insert(0, "row_id", rows["row_id"])
        features.to_parquet(directory / f"meta_{name}_{encoding}.parquet", index=False)


def load_matrices(name, adduct_encoding="onehot", meta=True, directory=DATA_PROCESSED):
    """Charge un sous-ensemble : (X, rows, noms des colonnes de X).

    - meta=True : X = [spectres binés | features de métadonnées], pour le MLP ;
    - meta=False : X = spectres binés seuls, pour la similarité cosinus.
    rows (row_id, inchikey14) est aligné sur les lignes de X. La colonne bin_k
    correspond aux m/z de k × 0,1 à (k + 1) × 0,1.
    """
    path = directory / f"X_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancez `python -m casmi.matrices`.")
    X = sp.load_npz(path).tocsr()
    rows = pd.read_parquet(directory / f"rows_{name}.parquet")
    names = [f"bin_{k}" for k in range(X.shape[1])]
    if meta:
        features = pd.read_parquet(directory / f"meta_{name}_{adduct_encoding}.parquet")
        if not np.array_equal(features.pop("row_id").to_numpy(), rows["row_id"].to_numpy()):
            raise ValueError(f"meta_{name}_{adduct_encoding}.parquet n'est pas aligné sur rows_{name}.parquet")
        X = sp.hstack([X, sp.csr_matrix(features.to_numpy(dtype=np.float32))], format="csr")
        names += features.columns.tolist()
    return X, rows, names


def load_encoder(adduct_encoding="onehot", directory=DATA_PROCESSED):
    """L'encodeur de métadonnées appris sur le split train, pour encoder d'autres spectres."""
    with open(directory / f"meta_encoder_{adduct_encoding}.pkl", "rb") as f:
        return pickle.load(f)


def build_matrices(directory=DATA_PROCESSED, n_train_sample=N_TRAIN_SAMPLE, seed=SEED):
    """Construit et sauvegarde les matrices de dev et de train_sample."""
    meta = load_clean_meta()[["row_id", "inchikey14", "keep", "query_ok"]]
    meta = meta.merge(load_split(), on="inchikey14", how="left")
    inputs = load_meta(META_INPUT_COLUMNS + ["num_peaks"])
    # Une ligne par row_id, dans l'ordre : la ligne i est le spectre i
    assert np.array_equal(meta["row_id"].to_numpy(), np.arange(len(meta)))
    assert np.array_equal(inputs["row_id"].to_numpy(), np.arange(len(inputs)))

    directory.mkdir(parents=True, exist_ok=True)
    is_train = ((meta["split"] == "train") & meta["keep"]).to_numpy()
    encoders = {}
    for encoding in ADDUCT_ENCODINGS:
        encoders[encoding] = fit_meta_encoder(inputs[is_train], adduct_encoding=encoding, seed=seed)
        with open(directory / f"meta_encoder_{encoding}.pkl", "wb") as f:
            pickle.dump(encoders[encoding], f)

    for name, row_ids in select_rows(meta, n_train_sample, seed).items():
        spectra = load_spectra(row_ids)
        assert np.array_equal(spectra["row_id"].to_numpy(), row_ids)
        X = bin_spectra(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"])
        metas = {encoding: meta_features(inputs.iloc[row_ids], e) for encoding, e in encoders.items()}
        save_subset(name, meta.iloc[row_ids], X, metas, directory)


if __name__ == "__main__":
    build_matrices()
    for name in ["dev", "train_sample"]:
        X, rows, names = load_matrices(name)
        size = sum((DATA_PROCESSED / f).stat().st_size for f in [f"X_{name}.npz", f"rows_{name}.parquet"])
        print(f"{name:13s}: {X.shape[0]:>7,} spectres, {rows['inchikey14'].nunique():>7,} molécules, "
              f"{X.shape[1]:,} colonnes, {X.nnz / X.shape[0]:.0f} valeurs non nulles par ligne, "
              f"{size / 1e6:.0f} Mo sur le disque")
