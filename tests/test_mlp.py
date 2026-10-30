import numpy as np
import pytest

torch = pytest.importorskip("torch")

from casmi.models.mlp import (  # noqa: E402
    SpectrumToFingerprint,
    TrainConfig,
    fingerprint_metrics,
    load_model,
    predict_proba,
    save_model,
    train_mlp,
)


def random_data(n=64, n_inputs=100, n_bits=256, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_inputs), dtype=np.float32)
    Y = (rng.random((n, n_bits)) < density).astype(np.float32)
    return X, Y


def test_forward_shape():
    model = SpectrumToFingerprint(n_inputs=100, hidden=32, n_bits=2048)
    assert model(torch.zeros(5, 100)).shape == (5, 2048)


def test_can_overfit_a_tiny_dataset():
    # Contrôle du plan : si le modèle ne sur-apprend pas un tout petit échantillon, il y a un bug
    X, Y = random_data()
    config = TrainConfig(hidden=256, dropout=0.0, lr=1e-3, batch_size=16, max_epochs=150, n_threads=2)
    model, history = train_mlp(X, Y, config=config, verbose=False)

    assert history[-1]["train_loss"] < history[0]["train_loss"] / 10
    assert fingerprint_metrics(predict_proba(model, X), Y)["tanimoto"] > 0.95


def test_early_stopping_keeps_best_epoch():
    # Sur des données aléatoires, un modèle qui apprend vite finit par apprendre le train
    # par cœur : la loss de validation remonte, et l'arrêt anticipé doit se déclencher
    X, Y = random_data(n=200)
    config = TrainConfig(hidden=256, dropout=0.0, lr=1e-2, batch_size=16, max_epochs=100, patience=3, n_threads=2)
    model, history = train_mlp(X[:150], Y[:150], X[150:], Y[150:], config=config, verbose=False)

    assert len(history) < 100
    best = min(h["val_loss"] for h in history)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        final = loss_fn(model(torch.as_tensor(X[150:])), torch.as_tensor(Y[150:])).item()
    assert final == pytest.approx(best, rel=1e-4)


def test_predict_proba_range():
    X, Y = random_data(n=10)
    model, _ = train_mlp(X, Y, config=TrainConfig(hidden=16, max_epochs=1, n_threads=1), verbose=False)
    probs = predict_proba(model, X)
    assert probs.shape == Y.shape
    assert probs.min() >= 0 and probs.max() <= 1


def test_fingerprint_metrics_hand_computed():
    probs = np.array([[0.9, 0.8, 0.1, 0.0], [0.6, 0.0, 0.0, 0.7]])
    truth = np.array([[1, 0, 1, 0], [1, 0, 0, 1]])
    m = fingerprint_metrics(probs, truth)
    # Ligne 1 : prédit {0, 1}, vrai {0, 2} → 1 commun / 3 → 1/3 ; ligne 2 : identique → 1
    assert m["tanimoto"] == pytest.approx((1 / 3 + 1) / 2)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)


def test_save_and_load(tmp_path):
    X, Y = random_data(n=20)
    config = TrainConfig(hidden=16, max_epochs=2, n_threads=1)
    model, _ = train_mlp(X, Y, config=config, verbose=False)

    save_model(model, config, tmp_path / "models" / "mlp.pt")
    loaded, loaded_config = load_model(tmp_path / "models" / "mlp.pt")

    assert loaded_config == config
    np.testing.assert_allclose(predict_proba(loaded, X), predict_proba(model, X), rtol=1e-6)
