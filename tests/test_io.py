from casmi.io import load_spectra


def test_load_spectra():
    assert load_spectra([0, 5])["row_id"].tolist() == [0, 5]