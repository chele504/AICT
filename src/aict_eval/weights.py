from __future__ import annotations

import numpy as np


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    denom = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    return (x - x_min) / denom


def grey_relational_analysis(
    features: np.ndarray,
    target: np.ndarray,
    distinguishing_coefficient: float = 0.5,
) -> np.ndarray:
    """
    计算各指标与目标序列的灰色关联度，输出归一化权重。
    """
    x = _normalize_minmax(features)
    y = _normalize_minmax(target.reshape(-1, 1)).reshape(-1, 1)
    diff = np.abs(x - y)
    min_diff = diff.min()
    max_diff = diff.max()
    coeff = (min_diff + distinguishing_coefficient * max_diff) / (
        diff + distinguishing_coefficient * max_diff + 1e-8
    )
    relation = coeff.mean(axis=0)
    return relation / (relation.sum() + 1e-8)


def coefficient_of_variation_weights(features: np.ndarray) -> np.ndarray:
    """
    依据变异系数进行客观赋权，方差越大说明区分度越强。
    """
    x = np.asarray(features, dtype=float)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    cv = std / (np.abs(mean) + 1e-8)
    return cv / (cv.sum() + 1e-8)


def combine_gra_cv_weights(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    融合 GRA 与 CV 权重，alpha 越大越偏向关联性。
    """
    gra_weights = grey_relational_analysis(features, target)
    cv_weights = coefficient_of_variation_weights(features)
    weights = alpha * gra_weights + (1.0 - alpha) * cv_weights
    return weights / (weights.sum() + 1e-8)


def estimate_gra_cv_alpha(
    features: np.ndarray,
    target: np.ndarray,
    min_alpha: float = 0.2,
    max_alpha: float = 0.8,
) -> float:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        return float(np.clip(0.5, min_alpha, max_alpha))
    y_std = float(np.std(y))
    if y_std < 1e-8:
        return float(np.clip(0.5, min_alpha, max_alpha))
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
