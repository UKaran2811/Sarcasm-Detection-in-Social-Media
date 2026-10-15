"""Metrics, bootstrap confidence intervals and McNemar's significance test."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics(y, pred, score=None):
    m = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Macro-F1": f1_score(y, pred, average="macro"),
    }
    if score is not None:
        m["ROC-AUC"] = roc_auc_score(y, score)
    return m


def bootstrap_f1_ci(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[idx], pred[idx], zero_division=0))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar test. b = A right & B wrong, c = A wrong & B right."""
    y, a, b_ = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    b = int(((a == y) & (b_ != y)).sum())
    c = int(((a != y) & (b_ == y)).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, binomtest(min(b, c), b + c, 0.5).pvalue
