import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def ks(y_true, y_score) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)

def gini(y_true, y_score) -> float:
    return float(2 * roc_auc_score(y_true, y_score) - 1)

def psi(expected, actual, bins: int = 10) -> float:
    expected, actual = np.asarray(expected), np.asarray(actual)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))