"""
Dimensionality reduction (UMAP or PCA) and clustering (HDBSCAN or k-means)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import umap
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN


def run_clustering(
    feature_matrix: np.ndarray,
    *,
    umap_n_neighbors: int = 10,
    umap_min_dist: float = 0.0,
    method: str = "hdbscan",
    k: int = 2,
    random_state: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    labels : np.ndarray
        1-D cluster labels (−1 for noise in HDBSCAN).
    xy : np.ndarray
        2-D UMAP coordinates, shape (n_samples, 2).
    """
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        random_state=random_state
    )
    f_len=feature_matrix[0].shape[1]
    features_array = np.array(feature_matrix).reshape(len(feature_matrix),f_len)
    # X = [f.flatten() for f in feature_matrix]

    xy = reducer.fit_transform(features_array)

    if method == "hdbscan":
        clusterer = HDBSCAN(min_samples=10, min_cluster_size=12)
        labels = clusterer.fit_predict(xy)
        
    elif method == "kmeans":
        if k is None:
            raise ValueError("k must be specified when method='kmeans'")
        clusterer = KMeans(n_clusters=k, random_state=random_state)
        labels = clusterer.fit_predict(xy)
    else:
        raise ValueError("method must be 'hdbscan' or 'kmeans'")

    return labels, xy