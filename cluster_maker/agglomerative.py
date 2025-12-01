
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def agglomerative(X, k, linkage="ward", metric="euclidean", compute_distances=True):
    """
    Perform agglomerative hierarchical clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    k : int
        Number of clusters
    linkage : str
        Linkage criterion ('ward', 'complete', 'average', 'single')
    metric : str
        Metric used to compute distances (replaces 'affinity'). Ignored if linkage='ward'.
    compute_distances : bool
        Whether to compute distances between clusters (needed for dendrograms)

    Returns
    -------
    labels : np.ndarray
        Cluster assignments for each sample
    None : placeholder for compatibility with kmeans API
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # For ward linkage, metric must be euclidean
    if linkage == "ward":
        metric = "euclidean"

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric=metric,
        compute_distances=compute_distances
    )

    labels = model.fit_predict(X)

    # Return labels and None to maintain API similarity with kmeans
    return labels, None