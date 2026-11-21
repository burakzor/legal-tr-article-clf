"""Linear SVM on TF-IDF features (scikit-learn LinearSVC).

One-vs-rest, squared hinge loss, L2 penalty. C is the inverse regularization strength.
The model has no probabilities: its scores are decision values (signed distances to each
class's hyperplane), so reports are written with probabilistic=False.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC


def fit(X: sp.csr_matrix, y: np.ndarray, C: float, class_weight: str, max_iter: int, seed: int) -> LinearSVC:
    """class_weight: 'balanced' (n / (n_classes * count_c)) or 'none'."""
    model = LinearSVC(C=C, class_weight=None if class_weight == "none" else class_weight,
                      max_iter=max_iter, random_state=seed)
    return model.fit(X, y)


def decision_scores(model: LinearSVC, X: sp.csr_matrix, n_classes: int) -> np.ndarray:
    """(n, n_classes) decision values; a class missing from training gets -inf so it is never predicted."""
    out = np.full((X.shape[0], n_classes), -np.inf)
    out[:, model.classes_] = model.decision_function(X)
    return out


def predict(model: LinearSVC, X: sp.csr_matrix, n_classes: int) -> np.ndarray:
    return decision_scores(model, X, n_classes).argmax(axis=1)
