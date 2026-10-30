import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi.mass import compute_neutral_masses, element_mass, parse_adduct

# Valeurs attendues calculées à la main pour precursor_mz = 500, avec
# m(H) = 1,00782503223 ; m(O) = 15,99491461957 ; m(Na) = 22,989769282 ;
# m(C) = 12 ; m_e = 0,000548579909 (en Da).
HAND_COMPUTED = [
    ("[M+H]+", 498.992724),        # 500 − m(H) + m_e
    ("[M-H]-", 501.007276),        # 500 + m(H) − m_e
    ("[M]+", 500.000549),          # 500 + m_e
    ("[M+Na]+", 477.010779),       # 500 − m(Na) + m_e
    ("[2M+Na]+", 238.505390),      # (500 − m(Na) + m_e) / 2
    ("[M+2H]2+", 997.985447),      # 2 × 500 − 2 m(H) + 2 m_e
    ("[M-2H]2-", 1002.014553),     # 2 × 500 + 2 m(H) − 2 m_e
    ("[M-H2O+H]+", 517.003288),    # 500 + m(H2O) − m(H) + m_e
    ("[M-2H2O+H]+", 535.013853),   # 500 + 2 m(H2O) − m(H) + m_e
    ("[M+CH2O2-H]-", 455.001797),  # 500 − m(CH2O2) + m(H) − m_e  (formiate)
]

INVALID = [
    "[Cat]2+",   # "Ca" suivi d'un "t" : ne doit pas être lu comme du calcium
    "[M+H]",     # pas de charge
    "M+H+",      # pas de crochets
    "[M+h]+",    # symbole en minuscule
    "[M+Xx]+",   # élément inconnu
    "[M+D]+",    # deutérium : RDKit le lit comme H, on le refuse
    "[0M+H]+",   # zéro molécule
    "[M+H]0+",   # charge nulle
    "[M+0H]+",   # coefficient nul
    "[M++H]+",
    "[M+H]++",
    " [M+H]+",
    "",
    None,
    np.nan,
]


@pytest.mark.parametrize("adduct, expected", HAND_COMPUTED)
def test_neutral_mass_hand_computed(adduct, expected):
    df = pd.DataFrame({"adduct": [adduct], "precursor_mz": [500.0]})
    assert compute_neutral_masses(df)[0] == pytest.approx(expected, abs=1e-6)


def test_parse_adduct_components():
    n_mol, delta, z = parse_adduct("[2M+Na-2H]-")
    assert n_mol == 2
    assert z == -1
    assert delta == pytest.approx(22.989769282 - 2 * 1.00782503223, abs=1e-8)


@pytest.mark.parametrize("adduct", INVALID)
def test_parse_adduct_rejects_invalid(adduct):
    with pytest.raises(ValueError):
        parse_adduct(adduct)


@pytest.mark.parametrize("adduct", INVALID)
def test_neutral_mass_invalid_is_nan(adduct):
    df = pd.DataFrame({"adduct": [adduct], "precursor_mz": [500.0]})
    assert np.isnan(compute_neutral_masses(df)[0])


def test_element_mass():
    assert element_mass("H") == pytest.approx(1.00782503223, abs=1e-8)
    assert element_mass("Cl") == pytest.approx(34.968852682, abs=1e-8)
    with pytest.raises(ValueError):
        element_mass("T")


def test_compute_neutral_masses_keeps_row_order():
    # Index désordonné, adducts répétés, une valeur manquante et un adduct invalide
    df = pd.DataFrame(
        {
            "adduct": ["[M-H]-", "[M+H]+", None, "[Cat]2+", "[M+H]+"],
            "precursor_mz": [500.0, 500.0, 500.0, 500.0, 300.0],
        },
        index=[42, 7, 3, 0, 15],
    )
    masses = compute_neutral_masses(df)

    assert isinstance(masses, np.ndarray)
    assert masses.shape == (5,)
    np.testing.assert_allclose(masses[[0, 1, 4]], [501.007276, 498.992724, 298.992724], atol=1e-6)
    assert np.isnan(masses[2]) and np.isnan(masses[3])


def test_neutral_mass_matches_rdkit_exact_mass():
    # Caféine en [M+H]+ : m/z = M + m(proton), avec m(proton) = 1,007276466621 Da (CODATA)
    exact_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C"))
    df = pd.DataFrame({"adduct": ["[M+H]+"], "precursor_mz": [exact_mass + 1.007276466621]})
    assert compute_neutral_masses(df)[0] == pytest.approx(exact_mass, abs=1e-6)
