"""Counterfactual-appropriate evaluation.

Math notes
----------
We never see both potential outcomes for one user, so AUC/accuracy are
meaningless for uplift. Instead, rank users by predicted uplift and measure
incremental conversions when targeting the top fraction:

  Qini(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k)

i.e. conversions among the treated top-k minus control conversions scaled to
the treated population size. AUUC = area under this curve.
"""
import numpy as np

_trap = getattr(np, "trapezoid", None) or np.trapz


def qini_curve(y, treatment, uplift_scores, n_bins: int = 100):
    order = np.argsort(-np.asarray(uplift_scores))
    y, t = np.asarray(y)[order], np.asarray(treatment)[order]
    n = len(y)
    fracs, qini = [0.0], [0.0]
    for i in np.linspace(n / n_bins, n, n_bins).astype(int):
        yt, tt = y[:i], t[:i]
        nt, nc = tt.sum(), (1 - tt).sum()
        if nt == 0 or nc == 0:
            continue
        fracs.append(i / n)
        qini.append(yt[tt == 1].sum() - yt[tt == 0].sum() * (nt / nc))
    return np.array(fracs), np.array(qini)


def auuc(y, treatment, uplift_scores) -> float:
    fracs, qini = qini_curve(y, treatment, uplift_scores)
    return float(_trap(qini, fracs))


def qini_coefficient(y, treatment, uplift_scores) -> float:
    """Area between the model curve and the random-targeting diagonal,
    per targeted user (scale-free across sample sizes)."""
    fracs, qini = qini_curve(y, treatment, uplift_scores)
    random_area = qini[-1] / 2  # straight line from (0,0) to (1, total incremental)
    coef = (float(_trap(qini, fracs)) - random_area) / len(y)
    print(f"[qini] AUUC = {_trap(qini, fracs):.2f}, total incremental = "
          f"{qini[-1]:.1f}, Qini coefficient = {coef:.6f}")
    return coef


def uplift_at_k(y, treatment, uplift_scores, k: float = 0.1) -> float:
    """Observed uplift (treated rate - control rate) inside the top-k slice."""
    n = len(y)
    idx = np.argsort(-np.asarray(uplift_scores))[: int(n * k)]
    yt, tt = np.asarray(y)[idx], np.asarray(treatment)[idx]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return float("nan")
    val = yt[tt == 1].mean() - yt[tt == 0].mean()
    print(f"[uplift@{k:.0%}] observed uplift in top slice = {val:+.5f} "
          f"(vs overall ATE -- bigger is better targeting)")
    return float(val)