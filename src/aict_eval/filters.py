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
    wavelet_wavelet: str = "db4"
    wavelet_mode: str = "soft"
    wavelet_level: int = 2
    sg_window_length: int = 5
    sg_polyorder: int = 2
    median_window: int = 5


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


def median_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    series = np.asarray(values, dtype=float).copy()
    n = series.size
    if n == 0:
        return series
    w = max(int(window), 3)
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.empty_like(series, dtype=float)
    padded = np.pad(series, (half, half), mode="edge")
    for i in range(n):
        out[i] = float(np.median(padded[i : i + w]))
    return out


def moving_average_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    series = np.asarray(values, dtype=float).copy()
    n = series.size
    if n == 0:
        return series
    w = max(int(window), 1)
    half = w // 2
    cumsum = np.cumsum(np.pad(series, (1, 0), mode="constant", constant_values=0.0))
    uniform = (cumsum[w:] - cumsum[:-w]) / float(w)
    if w > n:
        return np.full(n, float(np.mean(series)), dtype=float)
    out = np.empty(n, dtype=float)
    out[:half] = uniform[0]
    tail_start = max(0, half - (w - 1 - half))
    out[half : half + uniform.size - half + tail_start] = uniform[: n - half]
    out[half + uniform.size - half + tail_start :] = uniform[-1]
    out = out[:n]
    return out


def savitzky_golay_smooth(
    values: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    series = np.asarray(values, dtype=float).copy()
    n = series.size
    if n == 0:
        return series
    w = max(int(window_length), int(polyorder) + 2)
    if w % 2 == 0:
        w += 1
    if n < w:
        w = n if n % 2 == 1 else max(3, n - 1)
    p = min(int(polyorder), w - 1)
    half = w // 2
    indices = np.arange(w) - half
    vandermonde = np.vander(indices, N=p + 1, increasing=True)
    q, _ = np.linalg.qr(vandermonde, mode="reduced")
    kernel = q[:, 0] @ q[:, 0].T if False else np.linalg.pinv(vandermonde)[0]
    out = np.convolve(series, kernel[::-1], mode="same")
    if half > 0:
        left_pad = np.pad(series, (w, 0), mode="edge")
        right_pad = np.pad(series, (0, w), mode="edge")
        for i in range(half):
            local = left_pad[w + i - half : w + i + half + 1]
            local_vander = np.vander(np.arange(-half, half + 1), N=p + 1, increasing=True)
            coeff = np.linalg.lstsq(local_vander, local, rcond=None)[0]
            out[i] = coeff[0]
        for i in range(n - half, n):
            j = i - (n - w)
            local = right_pad[j : j + w]
            local_vander = np.vander(np.arange(-half, half + 1), N=p + 1, increasing=True)
            coeff = np.linalg.lstsq(local_vander, local, rcond=None)[0]
            out[i] = coeff[0]
    return out


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def wavelet_denoise_haar(
    values: np.ndarray,
    level: int = 2,
    mode: str = "soft",
    sigma: Optional[float] = None,
) -> np.ndarray:
    series = np.asarray(values, dtype=float).copy()
    n = series.size
    if n == 0:
        return series
    L = max(1, int(level))
    orig_n = n
    pad = 0
    required = 2**L
    if n % required != 0:
        pad = required - (n % required)
        series = np.pad(series, (0, pad), mode="edge")
        n = series.size

    coeffs_approx = [series.astype(float)]
    coeffs_detail = []
    current = series.astype(float)
    for _ in range(L):
        length = current.size
        if length < 2:
            break
        even = current[0:length:2]
        odd = current[1:length:2]
        approx = (even + odd) / np.sqrt(2.0)
        detail = (even - odd) / np.sqrt(2.0)
        coeffs_detail.append(detail)
        current = approx
        coeffs_approx.append(current)

    thresholded_details = []
    for d in coeffs_detail:
        if d.size == 0:
            thresholded_details.append(d)
            continue
        if sigma is None:
            mad = float(np.median(np.abs(d - np.median(d)))) / 0.6745
            thresh = mad * np.sqrt(2.0 * np.log(max(d.size, 2)))
        else:
            thresh = float(sigma)
        if mode.lower() == "soft":
            thresholded_details.append(_soft_threshold(d, thresh))
        else:
            mask = np.abs(d) >= thresh
            thresholded_details.append(d * mask.astype(float))

    current = coeffs_approx[-1]
    for detail in reversed(thresholded_details):
        reconstructed = np.zeros(current.size * 2, dtype=float)
        reconstructed[0::2] = (current + detail) / np.sqrt(2.0)
        reconstructed[1::2] = (current - detail) / np.sqrt(2.0)
        current = reconstructed
    out = current[:orig_n]
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
    if method in {"median", "medfilt"}:
        return median_smooth(values, window=params.median_window)
    if method in {"ma", "moving_average", "movingaverage"}:
        return moving_average_smooth(values, window=params.ema_window)
    if method in {"sg", "savgol", "savitzky_golay", "savitzky-golay"}:
        return savitzky_golay_smooth(
            values,
            window_length=params.sg_window_length,
            polyorder=params.sg_polyorder,
        )
    if method in {"wavelet", "haar", "dwt"}:
        return wavelet_denoise_haar(
            values,
            level=params.wavelet_level,
            mode=params.wavelet_mode,
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
