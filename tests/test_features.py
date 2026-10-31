import numpy as np
import pandas as pd
import pytest

from casmi.features import (
    META_INPUT_COLUMNS, adduct_groups, bin_index, bin_spectra, bin_spectrum, fit_meta_encoder, meta_features,
    n_bins, neutral_losses,
)

RAW = dict(transform="none", normalize=False)  # pour lire directement les intensités


def test_exemple_du_plan():
    # Cases de 10 Da : (12,1 ; 0,2) et (18,7 ; 1,0) dans la case 10-20, (24,2 ; 0,4) dans la case 20-30
    v = bin_spectrum([12.1, 18.7, 24.2], [0.2, 1.0, 0.4], bin_width=10, max_mz=100, **RAW)
    assert v.shape == (10,)
    assert v[:4].tolist() == pytest.approx([0, 1.0, 0.4, 0])


def test_pic_sur_une_frontiere():
    # Un pic à 20,0 va dans la case 20-30 ; 0,3 va bien dans la case 3 malgré 0.3 / 0.1 = 2.9999…
    assert bin_index([20.0], 10).tolist() == [2]
    assert bin_index([0.3], 0.1).tolist() == [3]
    v = bin_spectrum([20.0], [1.0], bin_width=10, max_mz=100, **RAW)
    assert np.flatnonzero(v).tolist() == [2]


def test_pic_au_dela_de_max_mz():
    v = bin_spectrum([50.0, 100.0, 150.0], [1.0, 1.0, 1.0], bin_width=10, max_mz=100, **RAW)
    assert np.flatnonzero(v).tolist() == [5]  # 100 et 150 sont ignorés


def test_spectre_vide():
    v = bin_spectrum([], [], bin_width=10, max_mz=100)
    assert v.shape == (10,) and not v.any()


def test_deux_pics_dans_la_meme_case():
    mzs, its = [12.0, 15.0], [0.3, 0.5]
    assert bin_spectrum(mzs, its, bin_width=10, max_mz=100, agg="max", **RAW)[1] == pytest.approx(0.5)
    assert bin_spectrum(mzs, its, bin_width=10, max_mz=100, agg="sum", **RAW)[1] == pytest.approx(0.8)


def test_pics_invalides_ignores():
    v = bin_spectrum([np.nan, -5.0, 30.0, 40.0, 50.0], [1.0, 1.0, np.nan, 0.0, 0.7], bin_width=10, max_mz=100, **RAW)
    assert np.flatnonzero(v).tolist() == [5]


def test_racine_et_normalisation():
    v = bin_spectrum([10.0, 20.0], [1.0, 0.25], bin_width=10, max_mz=100)
    # racine : 1 et 0,5 ; puis norme L2 = 1
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v[1] / v[2] == pytest.approx(2.0)
    assert v.dtype == np.float32


def test_matrice_creuse_identique_aux_vecteurs():
    spectra = [([12.1, 18.7, 24.2], [0.2, 1.0, 0.4]), ([], []), ([55.5, 55.6, 99.9], [1.0, 0.1, 0.3])]
    X = bin_spectra([s[0] for s in spectra], [s[1] for s in spectra], bin_width=10, max_mz=100)
    assert X.shape == (3, 10)
    for i, (mzs, its) in enumerate(spectra):
        assert X[i].toarray()[0] == pytest.approx(bin_spectrum(mzs, its, bin_width=10, max_mz=100))


def test_taille_par_defaut():
    assert n_bins() == 10_000
    assert n_bins(0.01, 1000) == 100_000


def test_pertes_neutres():
    losses, its = neutral_losses([50.0, 120.0, 181.07], [0.5, 1.0, 0.2], precursor_mz=181.07)
    # le pic du précurseur lui-même (perte nulle) est retiré
    assert losses.tolist() == pytest.approx([131.07, 61.07])
    assert its.tolist() == [0.5, 1.0]


# --- meta_features ---

def make_meta(n=60, seed=0):
    """Petit train factice : deux adducts fréquents, énergies observées en eV."""
    rng = np.random.default_rng(seed)
    positive = np.arange(n) % 3 != 0
    ce = rng.choice([20.0, 40.0, 60.0], n)
    return pd.DataFrame({
        "adduct": np.where(positive, "[M+H]+", "[M-H]-"),
        "ionization_mode": np.where(positive, "positive", "negative"),
        "instrument_type": "timsTOF Pro",
        "precursor_mz": rng.uniform(150, 600, n),
        "collision_energy_ev": [[v] for v in ce],
        "collision_energy_orig": [f"{v:g}" for v in ce],
        "collision_energy_orig_units": "eV",
        "num_peaks": rng.integers(5, 300, n),
    })


@pytest.fixture(scope="module")
def encoder():
    return fit_meta_encoder(make_meta(), min_adduct_count=10)


def test_meta_colonnes_fixes(encoder):
    names = meta_features(make_meta(), encoder).columns.tolist()
    assert names[:7] == ["mode_positif", "masse_neutre", "log_num_peaks", "ce_min", "ce_max", "ce_mean", "ce_manquante"]
    assert "adduct=[M+H]+" in names and "adduct=autre" in names and "instrument=Inconnu" in names
    # Un tableau d'une seule ligne a exactement les mêmes colonnes
    assert meta_features(make_meta().head(1), encoder).columns.tolist() == names


def test_meta_valeurs(encoder):
    df = make_meta().head(1).assign(adduct="[M+H]+", ionization_mode="positive", precursor_mz=181.0707,
                                    collision_energy_ev=[[20.0, 40.0]], num_peaks=100)
    row = meta_features(df, encoder).iloc[0]
    assert row["mode_positif"] == 1
    assert row["masse_neutre"] == pytest.approx((181.0707 - 1.00728) / 1000, abs=1e-6)
    assert row["log_num_peaks"] == pytest.approx(2)
    assert (row["ce_min"], row["ce_max"], row["ce_mean"]) == pytest.approx((0.2, 0.4, 0.3))
    assert row["ce_manquante"] == 0
    assert row["adduct=[M+H]+"] == 1 and row["instrument=timsTOF"] == 1
    assert row.dtype == np.float32


def test_meta_adduct_inconnu_va_dans_autre(encoder):
    df = make_meta().head(2).assign(adduct=["[M+Na]+", "[Cat]2+"])
    X = meta_features(df, encoder)
    assert X["adduct=autre"].tolist() == [1, 1]
    assert X.filter(like="adduct=").sum(axis=1).tolist() == [1, 1]  # une seule case à 1
    assert not X.isna().any().any()  # [Cat]2+ n'a pas de masse neutre : repli sur precursor_mz


def test_meta_instrument_manquant(encoder):
    df = make_meta().head(3).assign(instrument_type=[None, "nan", "Orbitrap Fusion"])
    X = meta_features(df, encoder)
    assert X["instrument=Inconnu"].tolist() == [1, 1, 0]
    assert X["instrument=Orbitrap/FT"].tolist() == [0, 0, 1]


def test_meta_energie_manquante(encoder):
    df = make_meta().head(2).assign(precursor_mz=[250.0, 250.0], collision_energy_ev=[None, None],
                                    collision_energy_orig=["35", None], collision_energy_orig_units=["NCE", "unknown"])
    X = meta_features(df, encoder)
    # Ligne 0 : règle NCE → eV (35 × 250 / 500 = 17,5 eV), considérée comme observée
    assert X.loc[0, "ce_mean"] == pytest.approx(0.175) and X.loc[0, "ce_manquante"] == 0
    # Ligne 1 : rien à reconstruire, l'imputeur prédit une valeur et l'indicatrice vaut 1
    assert X.loc[1, "ce_manquante"] == 1
    assert X.loc[1, "ce_min"] == X.loc[1, "ce_mean"] == X.loc[1, "ce_max"]
    assert 0.2 <= X.loc[1, "ce_mean"] <= 0.6


def test_meta_num_peaks_depuis_les_pics(encoder):
    df = make_meta().head(1).assign(num_peaks=10)
    X1 = meta_features(df, encoder)
    X2 = meta_features(df.drop(columns="num_peaks").assign(ms2_mzs=[np.arange(10.0)]), encoder)
    pd.testing.assert_frame_equal(X1, X2)


def test_meta_sans_fuite(encoder):
    # Seules les colonnes de META_INPUT_COLUMNS (et num_peaks) sont lues : les autres ne changent rien
    df = make_meta()
    extra = df.assign(normalized_smiles="CCO", inchikey14="X", precursor_error_ppm=99.0,
                      ingest_lib="gnps", base_peak_intensity=1e6)
    pd.testing.assert_frame_equal(meta_features(df[META_INPUT_COLUMNS + ["num_peaks"]], encoder),
                                  meta_features(extra, encoder))


# --- Adduct décomposé ---


def test_adduct_groups():
    assert adduct_groups("[M+H]+") == (1, 1, {"H": 1})
    assert adduct_groups("[M-H]-") == (1, -1, {"H": -1})
    assert adduct_groups("[2M+Na]+") == (2, 1, {"Na": 1})
    assert adduct_groups("[M+CH2O2-H]-") == (1, -1, {"CH2O2": 1, "H": -1})
    assert adduct_groups("[M-2H2O+H]+") == (1, 1, {"H2O": -2, "H": 1})
    assert adduct_groups("[M]+") == (1, 1, {})
    # Même groupe quelle que soit l'écriture ; un élément seul compte ses atomes
    assert adduct_groups("[M+NH4]+")[2] == adduct_groups("[3M+H4N]+")[2] == {"H4N": 1}
    assert adduct_groups("[M+Na2]2+") == (1, 2, {"Na": 2})
    assert adduct_groups("[M-H2]2-") == (1, -2, {"H": -2})
    assert adduct_groups("[Cat]2+") is None


@pytest.fixture(scope="module")
def encoder_decompose():
    train = make_meta()
    train.loc[:9, "adduct"] = "[M+Na]+"  # 10 spectres : Na est gardé avec min_adduct_count=10
    return fit_meta_encoder(train, adduct_encoding="decompose", min_adduct_count=10)


def test_decompose_colonnes(encoder, encoder_decompose):
    assert encoder_decompose.groups == ["H", "Na"]
    X = meta_features(make_meta(), encoder_decompose)
    adduct_cols = [c for c in X.columns if c.startswith("adduct")]
    assert adduct_cols == ["adduct_nM", "adduct_charge", "adduct_groupe=H", "adduct_groupe=Na",
                           "adduct_groupe=autre", "adduct_inconnu"]
    # Les autres features ne changent pas d'un encodage à l'autre
    X_onehot = meta_features(make_meta(), encoder)
    others = [c for c in X.columns if not c.startswith("adduct")]
    pd.testing.assert_frame_equal(X[others], X_onehot[others])


def test_decompose_valeurs(encoder_decompose):
    df = make_meta().head(5).assign(adduct=["[M-H]-", "[2M+Na]+", "[M+NH4]+", "[Cat]2+", None])
    X = meta_features(df, encoder_decompose).filter(like="adduct")
    assert X.iloc[0].tolist() == [1, -1, -1, 0, 0, 0]  # [M-H]-
    assert X.iloc[1].tolist() == [2, 1, 0, 1, 0, 0]    # [2M+Na]+
    assert X.iloc[2].tolist() == [1, 1, 0, 0, 1, 0]    # [M+NH4]+ : NH4 est rare ici, compté dans « autre »
    assert X.iloc[3].tolist() == [0, 0, 0, 0, 0, 1]    # non interprétable
    assert X.iloc[4].tolist() == [0, 0, 0, 0, 0, 1]    # manquant


def test_encodage_inconnu():
    with pytest.raises(ValueError):
        fit_meta_encoder(make_meta(), adduct_encoding="ordinal")
