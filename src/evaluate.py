"""Uplift evaluation: Qini curve and AUUC. Standard AUC does not apply —
we never observe both potential outcomes for the same user."""
import numpy as np


def qini_curve(y, treatment, uplift_scores, n_bins: int = 100):
    """Returns (fractions targeted, cumulative incremental conversions)."""
    order = np.argsort(-uplift_scores)
    y, t = np.asarray(y)[order], np.asarray(treatment)[order]
    n = len(y)
    fracs, qini = [0.0], [0.0]
    for i in np.linspace(n / n_bins, n, n_bins).astype(int):
        yt, tt = y[:i], t[:i]
        nt, nc = tt.sum(), (1 - tt).sum()
        if nt == 0 or nc == 0:
            continue
        incr = yt[tt == 1].sum() - yt[tt == 0].sum() * (nt / nc)
        fracs.append(i / n)
        qini.append(incr)
    return np.array(fracs), np.array(qini)


def auuc(y, treatment, uplift_scores) -> float:
    fracs, qini = qini_curve(y, treatment, uplift_scores)
    return float(np.trapezoid(qini, fracs))
