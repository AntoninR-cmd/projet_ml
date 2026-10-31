"""Transformer un spectre en vecteur de taille fixe : le binning.

L'axe m/z est découpé en cases régulières [k × bin_width, (k + 1) × bin_width[.
Chaque case reçoit l'intensité des pics qui y tombent (le maximum par défaut),
après une transformation qui donne plus de poids aux petits pics (racine carrée).
Le vecteur est ensuite normalisé (norme L2), ce qui prépare la similarité cosinus.

bin_spectrum traite un spectre et renvoie un vecteur dense ; bin_spectra traite
une liste de spectres d'un coup et renvoie une matrice creuse (scipy.sparse).
"""

import numpy as np
import scipy.sparse as sp

# Choix par défaut (voir notebooks/A/05_binning.ipynb) : 10 000 cases
BIN_WIDTH = 0.1  # Da
MAX_MZ = 1000.0  # Da ; les pics au-delà sont ignorés (1,3 % des pics du train, aucun du test visible)

TRANSFORMS = {
    "none": lambda x: x,
    "sqrt": np.sqrt,
    "log": lambda x: np.log1p(1000 * x) / np.log1p(1000),  # garde 1 → 1
}


def n_bins(bin_width=BIN_WIDTH, max_mz=MAX_MZ):
    """Nombre de cases du vecteur."""
    return int(np.ceil(round(max_mz / bin_width, 6)))


def bin_index(mz, bin_width=BIN_WIDTH):
    """Numéro de la case de chaque m/z (un pic sur une frontière va dans la case du dessus).

    L'arrondi évite les erreurs de calcul en virgule flottante : 0.3 / 0.1 vaut
    2.9999999999999996, et sans arrondi le pic tomberait dans la case 2 au lieu de 3.
    """
    return np.floor(np.round(np.asarray(mz, dtype=float) / bin_width, 6)).astype(np.int64)


def neutral_losses(mzs, intensities, precursor_mz):
    """Pertes neutres : precursor_mz − m/z de chaque fragment, avec la même intensité.

    Seules les pertes strictement positives sont gardées (un pic au-dessus du
    précurseur n'est pas un fragment). Se bine comme un spectre ordinaire.
    """
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    losses = precursor_mz - mzs
    keep = losses > 0
    return losses[keep], intensities[keep]


def bin_spectra(mzs_list, intensities_list, bin_width=BIN_WIDTH, max_mz=MAX_MZ,
                agg="max", transform="sqrt", normalize=True):
    """Bine une liste de spectres : matrice creuse CSR (n_spectres, n_bins), float32.

    - agg : "max" (le plus haut pic de la case) ou "sum" (la somme des pics) ;
    - transform : "sqrt", "log" ou "none", appliquée aux intensités avant l'agrégation ;
    - normalize : norme L2 de chaque ligne égale à 1 (les spectres vides restent à 0).

    Les pics invalides sont ignorés : m/z NaN, négatif ou ≥ max_mz, intensité NaN ou ≤ 0.
    """
    size = n_bins(bin_width, max_mz)
    lengths = np.array([len(m) for m in mzs_list], dtype=np.int64)
    rows = np.repeat(np.arange(len(lengths)), lengths)
    mz = np.concatenate([np.asarray(m, dtype=float) for m in mzs_list]) if len(rows) else np.empty(0)
    it = np.concatenate([np.asarray(i, dtype=float) for i in intensities_list]) if len(rows) else np.empty(0)

    valid = np.isfinite(mz) & (mz >= 0) & (mz < max_mz) & np.isfinite(it) & (it > 0)
    rows, mz, it = rows[valid], mz[valid], it[valid]
    cols = np.minimum(bin_index(mz, bin_width), size - 1)  # garde-fou contre l'arrondi en bord de vecteur
    values = TRANSFORMS[transform](it)

    # Agrégation par (spectre, case) : on trie les clés, puis on réduit chaque groupe
    keys = rows * size + cols
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], values[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.empty(0, dtype=np.int64)
    reduce = np.maximum.reduceat if agg == "max" else np.add.reduceat
    data = reduce(values, starts) if len(keys) else np.empty(0)
    keys = keys[starts]

    X = sp.csr_matrix((data.astype(np.float32), (keys // size, keys % size)), shape=(len(lengths), size))
    if normalize:
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        scale = np.divide(1, norms, out=np.zeros_like(norms), where=norms > 0)  # 0 pour un spectre vide
        X = sp.diags(scale.astype(np.float32)) @ X
    return X.tocsr()


def bin_spectrum(mzs, intensities, bin_width=BIN_WIDTH, max_mz=MAX_MZ, **options):
    """Bine un seul spectre : vecteur dense float32 de taille n_bins(bin_width, max_mz).

    Mêmes options que bin_spectra (agg, transform, normalize).
    """
    return bin_spectra([mzs], [intensities], bin_width, max_mz, **options).toarray()[0]
