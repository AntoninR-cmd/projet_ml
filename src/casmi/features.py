"""Transformer un spectre en vecteur de taille fixe : le binning.

L'axe m/z est découpé en cases régulières [k × bin_width, (k + 1) × bin_width[.
Chaque case reçoit l'intensité des pics qui y tombent (le maximum par défaut),
après une transformation qui donne plus de poids aux petits pics (racine carrée).
Le vecteur est ensuite normalisé (norme L2), ce qui prépare la similarité cosinus.

bin_spectrum traite un spectre et renvoie un vecteur dense ; bin_spectra traite
une liste de spectres d'un coup et renvoie une matrice creuse (scipy.sparse).

Les métadonnées utiles (mode, adduct, masse neutre, énergie de collision,
instrument, nombre de pics) sont encodées par meta_features, avec des encodages
appris sur le train seulement (fit_meta_encoder). L'adduct a deux encodages à
comparer en validation : one-hot, ou décomposé en nombre de M, charge et groupes.
"""

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingRegressor

from casmi.config import SEED
from casmi.mass import compute_neutral_masses, parse_adduct
from casmi.missing import INSTRUMENT_FAMILIES, ce_summary, instrument_family, nce_to_ev

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


# --- Features de métadonnées (voir notebooks/A/06_meta_features.ipynb) ---

# Colonnes lues par meta_features, toutes présentes dans le test ; s'y ajoute num_peaks,
# ou à défaut ms2_mzs (le test n'a pas de colonne num_peaks).
# Jamais de fuite (SMILES, inchikey, precursor_error_ppm), ni ingest_lib (absente du test),
# ni base_peak_intensity (proposée exclue, voir docs/dictionnaire_donnees.md).
META_INPUT_COLUMNS = [
    "adduct", "ionization_mode", "instrument_type", "precursor_mz",
    "collision_energy_ev", "collision_energy_orig", "collision_energy_orig_units",
]

MIN_ADDUCT_COUNT = 1_000  # 16 adducts gardés : 99,8 % du train, 100 % du test visible
ADDUCT_ENCODINGS = ["onehot", "decompose"]  # les deux encodages de l'adduct à comparer en A4.1
INSTRUMENTS = [name for name, _ in INSTRUMENT_FAMILIES] + ["Autre", "Inconnu"]

# Échelles fixes, qui ramènent les valeurs autour de 0-1 sans rien apprendre des données
MASS_SCALE = 1_000.0  # Da
CE_SCALE = 100.0  # eV

CE_MODEL_MAX_FIT = 300_000  # sous-échantillon d'entraînement de l'imputeur d'énergie


@dataclass
class MetaEncoder:
    """Ce que meta_features apprend du train.

    adducts et groups : les adducts et les groupes assez fréquents pour avoir leur colonne ;
    adduct_encoding : "onehot" ou "decompose" ; ce_model : l'imputeur d'énergie.
    """

    adducts: list
    groups: list
    adduct_encoding: str
    ce_model: HistGradientBoostingRegressor


_TERM_RE = re.compile(r"([+-])(\d*)([A-Za-z0-9]+)")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _hill(composition):
    """Formule brute dans l'ordre de Hill : C, H, puis les autres éléments par ordre alphabétique."""
    first = ["C", "H"] if "C" in composition else []
    order = first + sorted(set(composition) - set(first))
    return "".join(e + (str(composition[e]) if composition[e] > 1 else "") for e in order)


@lru_cache(maxsize=None)
def adduct_groups(adduct):
    """Décompose un adduct en (n_M, charge, {groupe: nombre}) ; None s'il n'est pas interprétable.

    Un groupe est un terme ajouté (nombre positif) ou perdu (nombre négatif) :
    "[M+CH2O2-H]-" -> (1, -1, {"CH2O2": 1, "H": -1}) ; "[2M+Na]+" -> (2, 1, {"Na": 1}).
    Un terme d'un seul élément compte ses atomes ("+Na2" = 2 Na, "-H2" = -2 H). Les autres
    restent des blocs (H2O, CH2O2…), écrits dans l'ordre de Hill : NH4 et H4N sont le même groupe.
    """
    try:
        n_mol, _, z = parse_adduct(adduct)
    except ValueError:
        return None
    inner = adduct[adduct.index("M") + 1:adduct.rindex("]")]
    groups = Counter()
    for sign, coef, formula in _TERM_RE.findall(inner):
        composition = Counter()
        for element, count in _ELEMENT_RE.findall(formula):
            composition[element] += int(count or 1)
        n = int(coef or 1) * (1 if sign == "+" else -1)
        if len(composition) == 1:
            [(name, count)] = composition.items()
            n *= count
        else:
            name = _hill(composition)
        groups[name] += n
    return n_mol, z, {g: n for g, n in groups.items() if n != 0}


def _adduct_decomposition(adducts, groups):
    """Une ligne par spectre : n_M, charge, nombre de chaque groupe gardé, autres groupes, adduct inconnu.

    adduct_groupe=autre compte les groupes rares (en valeur absolue) ; un adduct non
    interprétable a toutes ses colonnes à 0, sauf adduct_inconnu = 1.
    """
    adducts = adducts.fillna("")
    names = (["adduct_nM", "adduct_charge"] + [f"adduct_groupe={g}" for g in groups]
             + ["adduct_groupe=autre", "adduct_inconnu"])
    table = {}
    for adduct in adducts.unique():
        parsed = adduct_groups(adduct)
        if parsed is None:
            table[adduct] = [0] * (len(names) - 1) + [1]
            continue
        n_mol, z, found = parsed
        rare = sum(abs(n) for g, n in found.items() if g not in groups)
        table[adduct] = [n_mol, z] + [found.get(g, 0) for g in groups] + [rare, 0]
    table = pd.DataFrame.from_dict(table, orient="index", columns=names)
    return table.reindex(adducts.to_numpy()).set_axis(adducts.index)


def _num_peaks(df):
    return df["num_peaks"] if "num_peaks" in df else df["ms2_mzs"].map(len)


def _collision_energy(df):
    """ce_min, ce_max et ce_mean en eV ; les NA sont d'abord reconstruits par la règle NCE → eV."""
    ce = ce_summary(df["collision_energy_ev"])[["ce_min", "ce_max", "ce_mean"]]
    rule = nce_to_ev(df["collision_energy_orig"], df["collision_energy_orig_units"], df["precursor_mz"])
    fill = ce["ce_mean"].isna() & rule.notna()
    for col in ce:
        ce.loc[fill, col] = rule[fill]
    return ce


def _categories(df, adducts):
    """Mode, adduct et famille d'instrument, avec des catégories fixes.

    Un adduct absent de la liste apprise devient "autre" ; un instrument manquant, "Inconnu".
    """
    adduct = df["adduct"].where(df["adduct"].isin(adducts), "autre")
    instrument = instrument_family(df["instrument_type"]).astype("object").fillna("Inconnu")
    return pd.DataFrame({
        "mode_positif": (df["ionization_mode"] == "positive").astype(float),
        "adduct": pd.Categorical(adduct, categories=list(adducts) + ["autre"]),
        "instrument": pd.Categorical(instrument, categories=INSTRUMENTS),
    }, index=df.index)


def _ce_model_inputs(df, cats):
    """Entrées de l'imputeur d'énergie, les mêmes qu'en A1.2 (sans base_peak_intensity)."""
    return cats.assign(
        precursor_mz=df["precursor_mz"].to_numpy(),
        log_num_peaks=np.log10(np.maximum(_num_peaks(df).to_numpy(), 1)),
    )


def fit_meta_encoder(train, adduct_encoding="onehot", min_adduct_count=MIN_ADDUCT_COUNT, seed=SEED):
    """Apprend les encodages sur le train uniquement.

    - adduct_encoding : "onehot" (une colonne par adduct) ou "decompose" (n_M, charge et groupes) ;
    - adducts : ceux qui ont au moins min_adduct_count spectres ;
    - groups : les groupes présents dans au moins min_adduct_count spectres ;
    - ce_model : gradient boosting qui prédit l'énergie moyenne quand elle manque encore
      après la règle NCE → eV (l'imputeur retenu en A1.2).
    """
    if adduct_encoding not in ADDUCT_ENCODINGS:
        raise ValueError(f"adduct_encoding doit valoir {ADDUCT_ENCODINGS}, pas {adduct_encoding!r}")
    counts = train["adduct"].value_counts()
    adducts = counts.index[counts >= min_adduct_count].tolist()

    group_counts = Counter()
    for adduct, n in counts.items():
        parsed = adduct_groups(adduct)
        for group in (parsed[2] if parsed else []):
            group_counts[group] += n
    groups = [g for g, n in group_counts.most_common() if n >= min_adduct_count]

    X = _ce_model_inputs(train, _categories(train, adducts))
    ce = _collision_energy(train)["ce_mean"]
    idx = np.flatnonzero(ce.notna())
    if len(idx) > CE_MODEL_MAX_FIT:
        idx = np.random.default_rng(seed).choice(idx, CE_MODEL_MAX_FIT, replace=False)
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=seed)
    model.fit(X.iloc[idx], ce.iloc[idx])
    return MetaEncoder(adducts, groups, adduct_encoding, model)


def meta_features(df, encoder):
    """Features de métadonnées : DataFrame float32 aligné sur df, sans NA.

    Les noms des colonnes sont les noms des features. Les colonnes sont toujours les
    mêmes et dans le même ordre, quel que soit df (train, validation ou test).
    Seules les colonnes d'adduct dépendent de encoder.adduct_encoding ; l'imputeur
    d'énergie utilise les catégories d'adduct dans les deux cas.
    """
    cats = _categories(df, encoder.adducts)

    ce = _collision_energy(df)
    missing = ce["ce_mean"].isna()
    if missing.any():
        predicted = encoder.ce_model.predict(_ce_model_inputs(df, cats)[missing.to_numpy()])
        for col in ce:
            ce.loc[missing, col] = predicted

    # Adduct non interprétable : pas de masse neutre, on garde la masse de l'ion
    mass = pd.Series(compute_neutral_masses(df), index=df.index).fillna(df["precursor_mz"])

    out = pd.DataFrame({
        "mode_positif": cats["mode_positif"],
        "masse_neutre": mass / MASS_SCALE,
        "log_num_peaks": np.log10(np.maximum(_num_peaks(df), 1)),
        "ce_min": ce["ce_min"] / CE_SCALE,
        "ce_max": ce["ce_max"] / CE_SCALE,
        "ce_mean": ce["ce_mean"] / CE_SCALE,
        "ce_manquante": missing.astype(float),
    }, index=df.index)
    if encoder.adduct_encoding == "onehot":
        adduct = pd.get_dummies(cats["adduct"], prefix="adduct", prefix_sep="=")
    else:
        adduct = _adduct_decomposition(df["adduct"], encoder.groups)
    instrument = pd.get_dummies(cats["instrument"], prefix="instrument", prefix_sep="=")
    return pd.concat([out, adduct, instrument], axis=1).astype(np.float32)
