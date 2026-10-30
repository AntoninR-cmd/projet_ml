import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.config import TEST_PATH, TRAIN_PATH


PEAK_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities"]


def load_meta(columns=None):
    """Une ligne par spectre du train, sans les pics du spectre. Ajoute une colonne row_id.

    columns : liste de colonnes à lire (par défaut, toutes sauf les pics).
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_PATH} introuvable : téléchargez les données (voir README.md).")

    col_names = pq.ParquetFile(TRAIN_PATH).schema_arrow.names

    if columns is None:
        final_names = [c for c in col_names if c not in PEAK_COLUMNS]
    else:
        final_names = columns

    df_train = pd.read_parquet(TRAIN_PATH, columns=final_names)
    df_train.insert(0, "row_id", np.arange(len(df_train)))

    return df_train


def load_test(columns=None):
    """Le test complet, pics compris (le fichier est petit).

    columns : liste de colonnes à lire (par défaut, toutes).
    """
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"{TEST_PATH} introuvable : téléchargez les données (voir README.md).")

    df_test = pd.read_parquet(TEST_PATH, columns=columns)

    return df_test


def load_spectra(row_ids):
    """Les pics (PEAK_COLUMNS) des lignes demandées seulement, avec leur row_id.

    row_ids : liste (ou Series) de row_id, tels que renvoyés par load_meta.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_PATH} introuvable : téléchargez les données (voir README.md).")

    row_ids = np.asarray(row_ids, dtype=np.int64)

    # On ne lit que les blocs (row groups) qui contiennent les lignes demandées :
    # tout lire d'un coup demanderait plus de 12 Go de RAM.
    pf = pq.ParquetFile(TRAIN_PATH)
    if row_ids.min() < 0 or row_ids.max() >= pf.metadata.num_rows:
        raise IndexError(f"row_id hors de [0, {pf.metadata.num_rows - 1}]")

    # Numéro de la première ligne de chaque bloc : [0, 131072, 262144, ...]
    sizes = [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    # Bloc qui contient chaque row_id
    groups = np.searchsorted(starts, row_ids, side="right") - 1

    parts, positions = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)  # positions de ces lignes dans row_ids
        table = pf.read_row_group(g, columns=PEAK_COLUMNS)
        parts.append(table.take(row_ids[idx] - starts[g]).to_pandas())
        positions.append(idx)

    # Remettre les lignes dans l'ordre demandé
    df_spec = pd.concat(parts, ignore_index=True)
    df_spec = df_spec.iloc[np.argsort(np.concatenate(positions))].reset_index(drop=True)

    df_spec.insert(0, "row_id", row_ids)

    return df_spec
