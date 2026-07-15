from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DenoiseParams:
    method: str = "kalman"
    kalman_process_variance: float = 1e-4
    kalman_measurement_variance: float = 1e-2
    ema_alpha: float = 0.25
    ema_min_alpha: float = 0.05
    ema_max_alpha: float = 0.6
    ema_window: int = 5


def kalman_smooth_1d(
    values: np.ndarray,
    process_variance: float,
    measurement_variance: float,
) -> np.ndarray:
    series = np.asarray(values, dtype=float).copy()
    if series.size == 0:
        return series
    x_hat = float(series[0])
    p = 1.0
    q = float(process_variance)
    r = float(measurement_variance)
    out = np.empty_like(series, dtype=float)
    for i, z in enumerate(series):
        p = p + q
        k = p / (p + r)
        x_hat = x_hat + k * (float(z) - x_hat)
        p = (1.0 - k) * p
        out[i] = x_hat
    return out


def adaptive_ema_smooth(
    values: np.ndarray,
    alpha: float,
    min_alpha: float,
    max_alpha: float,
    window: int,
) -> np.ndarray:
    series = np.asarray(values, dtype=float).copy()
    if series.size == 0:
        return series
    w = max(int(window), 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    base_alpha = float(alpha)
    lo = float(min_alpha)
    hi = float(max_alpha)
    for i in range(1, series.size):
        start = max(0, i - w)
        local = series[start:i]
        std = float(np.std(local)) if local.size else 0.0
        scale = std / (std + 1.0)
        a = float(np.clip(base_alpha * (0.5 + scale), lo, hi))
        out[i] = a * series[i] + (1.0 - a) * out[i - 1]
    return out


def _denoise_series(values: np.ndarray, params: DenoiseParams) -> np.ndarray:
    method = params.method.lower().strip()
    if method == "kalman":
        return kalman_smooth_1d(
            values,
            process_variance=params.kalman_process_variance,
            measurement_variance=params.kalman_measurement_variance,
        )
    if method in {"ema", "adaptive_ema", "adaptive-ema"}:
        return adaptive_ema_smooth(
            values,
            alpha=params.ema_alpha,
            min_alpha=params.ema_min_alpha,
            max_alpha=params.ema_max_alpha,
            window=params.ema_window,
        )
    raise ValueError(f"未知去噪方法: {params.method}")


def denoise_dataframe(
    df: pd.DataFrame,
    columns: Iterable[str],
    params: DenoiseParams,
    group_column: Optional[str] = None,
    sort_column: Optional[str] = None,
) -> pd.DataFrame:
    columns = list(columns)
    if not columns:
        return df
    out = df.copy()

    if group_column:
        if group_column not in out.columns:
            raise ValueError(f"未找到去噪分组列: {group_column}")
        grouped = out.groupby(group_column, sort=False)
        pieces = []
        for _, frame in grouped:
            piece = frame.copy()
            if sort_column:
                if sort_column not in piece.columns:
                    raise ValueError(f"未找到去噪排序列: {sort_column}")
                piece = piece.sort_values(sort_column, kind="mergesort")
            for col in columns:
                piece[col] = _denoise_series(piece[col].to_numpy(), params)
            pieces.append(piece)
        return pd.concat(pieces, axis=0).sort_index()

    if sort_column:
        if sort_column not in out.columns:
            raise ValueError(f"未找到去噪排序列: {sort_column}")
        out = out.sort_values(sort_column, kind="mergesort")
        for col in columns:
            out[col] = _denoise_series(out[col].to_numpy(), params)
        return out.sort_index()

    for col in columns:
        out[col] = _denoise_series(out[col].to_numpy(), params)
    return out

