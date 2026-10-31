import numpy as np
import pytest

from casmi.features import bin_index, bin_spectra, bin_spectrum, n_bins, neutral_losses

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
