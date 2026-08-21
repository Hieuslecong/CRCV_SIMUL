from __future__ import annotations

from collections import defaultdict
import numpy as np


PRIMARY = ("dice", "crack_iou")


def paired_delta(base, refined):
    b = np.asarray(base, dtype=float)
    r = np.asarray(refined, dtype=float)
    if b.shape != r.shape or b.ndim != 1 or len(b) < 2:
        raise ValueError("base/refined must be same-length 1-D arrays with n>=2")
    if not (np.isfinite(b).all() and np.isfinite(r).all()):
        raise ValueError("metrics must be finite")
    return r - b


def bootstrap_ci(base, refined, seed=1337, n_boot=10000, alpha=0.05):
    """Naive paired IID bootstrap retained for diagnostics/backward compatibility.

    Publication claims with correlated crops/lineages must use cluster_bootstrap_ci.
    """
    d = paired_delta(base, refined)
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "method": "paired_iid_bootstrap",
    }


def _cluster_means(base, refined, clusters):
    d = paired_delta(base, refined)
    c = np.asarray(clusters, dtype=object)
    if c.ndim != 1 or len(c) != len(d):
        raise ValueError("clusters must be a same-length 1-D array")
    if any(x is None or str(x) == "" for x in c):
        raise ValueError("cluster ids must be non-empty")
    grouped = defaultdict(list)
    for delta, cluster in zip(d, c):
        grouped[str(cluster)].append(float(delta))
    if len(grouped) < 2:
        raise ValueError("at least two independent clusters are required")
    keys = sorted(grouped)
    means = np.asarray([np.mean(grouped[k]) for k in keys], dtype=float)
    return keys, means, d


def cluster_bootstrap_ci(base, refined, clusters, seed=1337, n_boot=10000, alpha=0.05):
    """Paired cluster bootstrap using lineage/parent as the sampling unit.

    Each cluster contributes one mean paired effect, so multiple crops/backbones
    from the same lineage cannot masquerade as independent samples.
    """
    keys, cluster_means, raw_d = _cluster_means(base, refined, clusters)
    rng = np.random.default_rng(seed)
    m = len(cluster_means)
    idx = rng.integers(0, m, size=(int(n_boot), m))
    boot = cluster_means[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(cluster_means.mean()),
        "raw_pair_mean": float(raw_d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_pairs": int(len(raw_d)),
        "n_clusters": int(m),
        "clusters": keys,
        "method": "paired_cluster_bootstrap",
    }


def paired_permutation_p(base, refined, seed=1337, n_perm=20000, alternative="greater"):
    d = paired_delta(base, refined)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), len(d)))
    null = (signs * d).mean(axis=1)
    if alternative == "greater":
        p = (np.count_nonzero(null >= obs) + 1) / (len(null) + 1)
    elif alternative == "two-sided":
        p = (np.count_nonzero(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    else:
        raise ValueError("alternative must be greater or two-sided")
    return {
        "mean_delta": obs,
        "p": float(p),
        "n_perm": int(n_perm),
        "alternative": alternative,
        "method": "paired_iid_sign_permutation",
    }


def cluster_permutation_p(base, refined, clusters, seed=1337, n_perm=20000, alternative="greater"):
    """Cluster-level sign-flip permutation test for paired effects."""
    _, cluster_means, _ = _cluster_means(base, refined, clusters)
    obs = float(cluster_means.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), len(cluster_means)))
    null = (signs * cluster_means).mean(axis=1)
    if alternative == "greater":
        p = (np.count_nonzero(null >= obs) + 1) / (len(null) + 1)
    elif alternative == "two-sided":
        p = (np.count_nonzero(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    else:
        raise ValueError("alternative must be greater or two-sided")
    return {
        "mean_delta": obs,
        "p": float(p),
        "n_perm": int(n_perm),
        "n_clusters": int(len(cluster_means)),
        "alternative": alternative,
        "method": "paired_cluster_sign_permutation",
    }


def holm_bonferroni(p_values):
    p = np.asarray(p_values, float)
    if p.ndim != 1 or len(p) == 0 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid p-values")
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()
