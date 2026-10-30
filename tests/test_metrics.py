import pytest

from casmi.metrics import mrr_at_k, recall_at_k


def test_mrr_calculated_by_hand():
    true_keys = ["A", "B", "C"]
    ranked_keys = [
        ["A", "X"],      # rang 1
        ["X", "B"],      # rang 2
        ["X", "Y"],      # absente
    ]

    assert mrr_at_k(true_keys, ranked_keys, k=25) == pytest.approx(0.5)


def test_recall_at_k():
    true_keys = ["A", "B", "C"]
    ranked_keys = [
        ["A"],
        ["X", "B"],
        ["X", "Y"],
    ]

    assert recall_at_k(true_keys, ranked_keys, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(true_keys, ranked_keys, k=2) == pytest.approx(2 / 3)


def test_empty_candidate_list_scores_zero():
    assert mrr_at_k(["A"], [[]], k=25) == 0.0
    assert recall_at_k(["A"], [[]], k=25) == 0.0


def test_none_candidate_keeps_its_position():
    assert mrr_at_k(["A"], [[None, "A"]], k=2) == pytest.approx(0.5)


def test_duplicate_candidate_keeps_its_position():
    assert mrr_at_k(["A"], [["X", "X", "A"]], k=3) == pytest.approx(1 / 3)


def test_mismatched_number_of_queries_is_rejected():
    with pytest.raises(ValueError):
        mrr_at_k(["A"], [], k=25)