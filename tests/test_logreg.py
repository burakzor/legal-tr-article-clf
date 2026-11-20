import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from legalclf.evaluation import bootstrap_macro_f1
from legalclf.features import truncate_words
from legalclf.models import logistic_regression as logreg


def test_truncate_words():
    texts = pd.Series(["bir iki üç dört", "tek"])
    assert truncate_words(texts, 2).tolist() == ["bir iki", "tek"]
    assert truncate_words(texts, None).tolist() == texts.tolist()


def test_sparse_and_dense_forward_match():
    model = logreg.LogisticRegression(5, 3)
    X = sp.random(4, 5, density=0.5, format="csr", random_state=0, dtype=np.float32)
    sparse_out = model(logreg.to_torch_sparse(X, "cpu"))
    dense_out = model(torch.from_numpy(X.toarray()))
    assert torch.allclose(sparse_out, dense_out, atol=1e-6)


def test_balanced_class_weights():
    w = logreg.class_weights(np.array([0, 0, 0, 1]), 2, "balanced")
    assert np.allclose(w, [4 / 6, 2.0])
    assert logreg.class_weights(np.array([0, 1]), 2, "none") is None


def test_fit_learns_separable_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=300)
    X = sp.csr_matrix(np.eye(3, dtype=np.float32)[y] + rng.random((300, 3), dtype=np.float32) * 0.1)
    model, history = logreg.fit(X, y, X, y, 3, lr=0.1, weight_decay=0.0, batch_size=32, max_epochs=20,
                                patience=3, class_weight="balanced", seed=0, device="cpu")
    assert (logreg.predict(model, X, "cpu") == y).mean() > 0.95
    assert history[0]["epoch"] == 1


def test_bootstrap_interval_contains_point_estimate():
    docs = pd.Series(np.repeat(np.arange(50), 4).astype(str))
    y = np.tile([0, 1, 2, 1], 50)
    pred = y.copy()
    pred[::7] = 0
    low, high = bootstrap_macro_f1(docs, y, pred, 3, n=200, seed=0)
    assert 0 <= low <= high <= 1
