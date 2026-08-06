from __future__ import annotations

from itertools import product

import numpy as np


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    denom = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    return (x - x_min) / denom


def _normalize_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    denom = np.where(std == 0, 1.0, std)
    return (x - mean) / denom


def grey_relational_analysis(
    features: np.ndarray,
    target: np.ndarray,
    distinguishing_coefficient: float = 0.5,
) -> np.ndarray:
    x = _normalize_minmax(features)
    y = _normalize_minmax(target.reshape(-1, 1)).reshape(-1, 1)
    diff = np.abs(x - y)
    min_diff = diff.min()
    max_diff = diff.max()
    coeff = (min_diff + distinguishing_coefficient * max_diff) / (
        diff + distinguishing_coefficient * max_diff + 1e-8
    )
    relation = coeff.mean(axis=0)
    relation = np.clip(relation, 1e-8, None)
    return relation / (relation.sum() + 1e-8)


def coefficient_of_variation_weights(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    cv = std / (np.abs(mean) + 1e-8)
    cv = np.clip(cv, 1e-8, None)
    return cv / (cv.sum() + 1e-8)


def pearson_correlation_weights(
    features: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    n_features = x.shape[1]
    weights = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        col = x[:, j]
        if float(np.std(col)) < 1e-8 or float(np.std(y)) < 1e-8:
            weights[j] = 1e-6
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        weights[j] = abs(corr) if np.isfinite(corr) else 1e-6
    weights = np.clip(weights, 1e-8, None)
    return weights / (weights.sum() + 1e-8)


def entropy_weights(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    x_norm = _normalize_minmax(x)
    x_sum = x_norm.sum(axis=0, keepdims=True)
    p = x_norm / (x_sum + 1e-12)
    p = np.clip(p, 1e-12, None)
    n = x.shape[0]
    log_p = np.log(p)
    e = -1.0 / max(1.0, np.log(n)) * (p * log_p).sum(axis=0)
    d = 1.0 - e
    d = np.clip(d, 1e-8, None)
    return d / (d.sum() + 1e-8)


def standard_deviation_weights(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    std = np.std(x, axis=0)
    std = np.clip(std, 1e-8, None)
    return std / (std.sum() + 1e-8)


def combine_gra_cv_weights(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.5,
    beta: float | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    gra_weights = grey_relational_analysis(features, target)
    cv_weights = coefficient_of_variation_weights(features)
    pearson_weights = pearson_correlation_weights(features, target)
    entropy_w = entropy_weights(features)

    a = float(np.clip(alpha, 0.0, 1.0))
    if beta is None:
        b = (1.0 - a) * 0.5
    else:
        b = float(np.clip(beta, 0.0, 1.0 - a))
    if gamma is None:
        g = (1.0 - a - b) * 0.6
    else:
        g = float(np.clip(gamma, 0.0, max(0.0, 1.0 - a - b)))
    d = max(0.0, 1.0 - a - b - g)

    weights = a * gra_weights + b * cv_weights + g * pearson_weights + d * entropy_w
    weights = np.clip(weights, 1e-10, None)
    return weights / (weights.sum() + 1e-10)


def _objective_alpha(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    w = combine_gra_cv_weights(features, target, alpha=alpha, beta=beta, gamma=gamma)
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    x_weighted = x * w.reshape(1, -1)
    score = x_weighted.sum(axis=1)
    if float(np.std(score)) < 1e-8 or float(np.std(y)) < 1e-8:
        return -1.0
    corr = float(np.corrcoef(score, y)[0, 1])
    if not np.isfinite(corr):
        return -1.0
    return abs(corr)


def estimate_gra_cv_alpha(
    features: np.ndarray,
    target: np.ndarray,
    min_alpha: float = 0.2,
    max_alpha: float = 0.8,
    grid_search: bool = True,
) -> float:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        return float(np.clip(0.5, min_alpha, max_alpha))
    y_std = float(np.std(y))
    if y_std < 1e-8:
        return float(np.clip(0.5, min_alpha, max_alpha))

    if not grid_search:
        corrs = []
        for j in range(x.shape[1]):
            col = x[:, j]
            if float(np.std(col)) < 1e-8:
                continue
            c = float(np.corrcoef(col, y)[0, 1])
            if np.isfinite(c):
                corrs.append(abs(c))
        score = float(np.mean(corrs)) if corrs else 0.0
        alpha = float(min_alpha + (max_alpha - min_alpha) * np.clip(score, 0.0, 1.0))
        return float(np.clip(alpha, min_alpha, max_alpha))

    alphas = np.linspace(min_alpha, max_alpha, 5)
    betas = np.linspace(0.05, 0.4, 4)
    gammas = np.linspace(0.05, 0.5, 4)
    best = -1.0
    best_alpha = 0.5
    for a, b, g in product(alphas, betas, gammas):
        if a + b + g > 1.0:
            continue
        obj = _objective_alpha(x, y, float(a), float(b), float(g))
        if obj > best:
            best = obj
            best_alpha = float(a)
    if best <= 0:
        corrs = []
        for j in range(x.shape[1]):
            col = x[:, j]
            if float(np.std(col)) < 1e-8:
                continue
            c = float(np.corrcoef(col, y)[0, 1])
            if np.isfinite(c):
                corrs.append(abs(c))
        score = float(np.mean(corrs)) if corrs else 0.0
        best_alpha = float(min_alpha + (max_alpha - min_alpha) * np.clip(score, 0.0, 1.0))
    return float(np.clip(best_alpha, min_alpha, max_alpha))
