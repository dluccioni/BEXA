"""Segmenting maps into regions: smoothness masks, k-means and DBSCAN clustering.

Ports of the ``Yue idea`` notebook cells and of v9's ``plot_smoothness_analysis``,
``plot_kmeans_clustering`` and ``plot_dbscan_clustering`` (the plotting lives in
:mod:`bexa.viz.maps`). scikit-learn is imported when a clustering function
is called.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bexa.core.backend import to_host

__all__ = ["cluster_stats", "dbscan_map", "kmeans_com", "kmeans_profiles", "smoothness_mask"]


def smoothness_mask(
    image: Any, window: int = 6, threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pixels whose local standard deviation is below a threshold (``mask, metric, threshold``).

    The metric is the population standard deviation of a ``(2h+1)`` window,
    ``h = window // 2``, evaluated for interior pixels and zero on the border,
    exactly as the notebook's sliding-window loop but computed with two
    uniform filters. ``threshold`` defaults to the mean of the non-zero metric.
    """
    from scipy.ndimage import uniform_filter

    data = np.asarray(to_host(image), dtype=np.float64)
    half = window // 2
    size = 2 * half + 1
    mean = uniform_filter(data, size, mode="constant")
    mean_sq = uniform_filter(data * data, size, mode="constant")
    metric = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
    interior = np.zeros_like(metric, dtype=bool)
    if half == 0:
        interior[:] = True
    elif data.shape[0] > 2 * half and data.shape[1] > 2 * half:
        interior[half:-half, half:-half] = True
    metric[~interior] = 0.0  # a window that does not fit leaves no interior, hence no metric
    if threshold is None:
        positive = metric[metric > 0]
        threshold = float(positive.mean()) if positive.size else 0.0
    return metric < threshold, metric, float(threshold)


def _scaled(features: np.ndarray, scale: bool) -> np.ndarray:
    if not scale:
        return features
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(features)


def kmeans_com(
    com_a: Any,
    com_b: Any,
    n_clusters: int = 5,
    features: str = "angle_magnitude",
    scale: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster pixels by their two COM values (``"raw"``) or by angle and magnitude.

    Returns a label map (NaN where a COM was NaN).
    """
    from sklearn.cluster import KMeans

    a = np.asarray(to_host(com_a), dtype=float)
    b = np.asarray(to_host(com_b), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if features == "angle_magnitude":
        columns = np.stack([np.arctan2(b[mask], a[mask]), np.hypot(a[mask], b[mask])], axis=-1)
    elif features == "raw":
        columns = np.stack([a[mask], b[mask]], axis=-1)
    else:
        raise ValueError(f"unknown features {features!r}; use angle_magnitude or raw")
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
        _scaled(columns, scale)
    )
    out = np.full(a.shape, np.nan)
    out[mask] = labels
    return out


def kmeans_profiles(
    volume: Any, n_clusters: int = 5, scale: bool = False, random_state: int = 42
) -> np.ndarray:
    """Cluster pixels by their whole rocking-curve profile (``volume`` is ``(frames, y, x)``)."""
    from sklearn.cluster import KMeans

    data = np.asarray(to_host(volume), dtype=float)
    profiles = data.reshape(data.shape[0], -1).T
    mask = ~np.any(np.isnan(profiles), axis=1)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
        _scaled(profiles[mask], scale)
    )
    out = np.full(profiles.shape[0], np.nan)
    out[mask] = labels
    return out.reshape(data.shape[1:])


def dbscan_map(com_a: Any, com_b: Any, eps: float = 2.0, min_samples: int = 5) -> np.ndarray:
    """DBSCAN on the standardised COM pair; ``-1`` marks noise, NaN missing pixels."""
    from sklearn.cluster import DBSCAN

    a = np.asarray(to_host(com_a), dtype=float)
    b = np.asarray(to_host(com_b), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    columns = np.stack([a[mask], b[mask]], axis=-1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(_scaled(columns, True))
    out = np.full(a.shape, np.nan)
    out[mask] = labels
    return out


def cluster_stats(labels: Any, maps: dict[str, Any]) -> dict[int, dict[str, dict[str, float]]]:
    """Mean, std, median, min and max of every map inside every cluster."""
    labels = np.asarray(to_host(labels), dtype=float)
    out: dict[int, dict[str, dict[str, float]]] = {}
    for value in np.unique(labels[np.isfinite(labels)]):
        region = labels == value
        stats: dict[str, dict[str, float]] = {}
        for name, data in maps.items():
            values = np.asarray(to_host(data), dtype=float)[region]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            stats[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "n": int(values.size),
            }
        out[int(value)] = stats
    return out
