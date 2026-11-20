"""Multinomial logistic regression on TF-IDF features, in PyTorch.

One linear layer + CrossEntropyLoss (softmax) = multinomial logistic regression.
L2 regularization is Adam's weight_decay. Training stops early on dev macro-F1.
TF-IDF batches stay sparse, so the 100k-feature input is never densified.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from legalclf.evaluation import scores


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits; softmax is applied inside the loss."""
        if x.is_sparse:
            return torch.sparse.mm(x, self.linear.weight.t()) + self.linear.bias
        return self.linear(x)


def to_torch_sparse(X: sp.csr_matrix, device: str) -> torch.Tensor:
    coo = X.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce().to(device)


def class_weights(y: np.ndarray, n_classes: int, mode: str) -> np.ndarray | None:
    """'balanced': n / (n_classes * count_c), as in scikit-learn; 'none': no weighting."""
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return len(y) / (n_classes * np.maximum(counts, 1))


@torch.no_grad()
def predict_proba(model: LogisticRegression, X: sp.csr_matrix, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(to_torch_sparse(X[i:i + batch_size], device)), dim=1).cpu().numpy()
           for i in range(0, X.shape[0], batch_size)]
    return np.vstack(out)


def predict(model: LogisticRegression, X: sp.csr_matrix, device: str) -> np.ndarray:
    return predict_proba(model, X, device).argmax(axis=1)


def fit(X_train: sp.csr_matrix, y_train: np.ndarray, X_dev: sp.csr_matrix, y_dev: np.ndarray, n_classes: int,
        lr: float, weight_decay: float, batch_size: int, max_epochs: int, patience: int,
        class_weight: str, seed: int, device: str) -> tuple[LogisticRegression, list[dict]]:
    """Train with Adam; return the model at the epoch with the best dev macro-F1, and the per-epoch history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LogisticRegression(X_train.shape[1], n_classes).to(device)
    w = class_weights(y_train, n_classes, class_weight)
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_train_t = torch.from_numpy(y_train).to(device)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for b in np.array_split(rng.permutation(X_train.shape[0]), max(1, X_train.shape[0] // batch_size)):
            opt.zero_grad()
            loss = loss_fn(model(to_torch_sparse(X_train[b], device)), y_train_t[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)

        dev = scores(y_dev, predict(model, X_dev, device), n_classes)
        history.append({"epoch": epoch, "train_loss": total / X_train.shape[0],
                        "dev_macro_f1": dev["macro_f1"], "dev_accuracy": dev["accuracy"]})
        if dev["macro_f1"] > best_f1:
            best_f1, bad = dev["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, history
