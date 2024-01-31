
import numpy as np
from anndata import AnnData
from spatialdata import SpatialData
import scipy.sparse as sp

from scanpy import logging as logg
from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _assert_spatial_basis,
    _save_data
)


def _nhood_enrichment(adj : sp.spmatrix, y : sp.spmatrix) -> np.ndarray:
    """
    Compute the neighborhood enrichment z-score.

    This function calculates the z-score for neighborhood enrichment, 
    a metric to assess the interaction between different types of points (e.g., cells)
    in spatial omics data based on their spatial adjacency and labels.

    Parameters:
    adj (sp.spmatrix): Adjacency matrix representing the spatial relationships between points. 
                       Each entry adj[i, j] is non-zero if point i is adjacent to point j.
    y (sp.spmatrix): Binary matrix representing the presence of specific labels for each point. 
                     Each row corresponds to a point, and each column corresponds to a label.

    Returns:
    np.ndarray: A tuple containing:
                - zscore: A matrix of z-scores, representing the strength of enrichment 
                          between each pair of labels.
                - total_neighbors_count: A matrix representing the total count of neighbors 
                                         for each cell type.

    The function computes the z-score by comparing the actual count of neighbors with 
    a specific label to the expected count under a random distribution of labels.
    """

    # Calculate the number of specific neighbors for each cell
    neighbors_count = adj @ y

    # Total count of specific neighbors for each cell type
    total_neighbors_count = y.T @ neighbors_count
    total_neighbors_count = total_neighbors_count.A  # Convert to dense array


    # Number of points (nodes)
    n = adj.shape[0]

    # Compute mean and variance
    mean_interaction = neighbors_count.sum(axis=0).A.flatten() / n

    # Squaring the data for variance calculation
    neighbors_count.data **= 2
    var_interaction = neighbors_count.sum(axis=0).A.flatten() / n - mean_interaction**2

    # Total count of points with specific label
    total_label_count = y.sum(axis=0).A.flatten()

    # Compute z-score for neighborhood enrichment
    zscore = (total_neighbors_count - np.outer(total_label_count, mean_interaction)) / np.sqrt(np.outer(total_label_count, var_interaction))

    # Transpose to match Squidpy    
    zscore = zscore.T
    total_label_count = total_label_count.T
    return zscore, total_neighbors_count




def nhood_enrichment(
    adata : AnnData | SpatialData,
    cluster_key : str,
    connectivity_key: str | None = None,
    spatial_key: str = Key.obsm.spatial,
    copy: bool = None,
    adj: sp.spmatrix | None = None
):
    """
    Computes the neighborhood enrichment z-scores.

    This function calculates the z-score for neighborhood enrichment, 
    a metric to assess the interaction between different types of points (e.g., cells)
    in spatial omics data based on their spatial adjacency and labels.

    This function serves as a wrapper around the function `_nhood_enrichment` to enable
    Squidpy-like usage.


    Parameters:
    adata (AnnData | SpatialData): The annotated data object, which can be either 
                                   AnnData or SpatialData.
    cluster_key (str): Key in adata.obs to use for clustering.
    connectivity_key (str | None, optional): Key in adata.obsp to define connectivity. 
                                             Defaults to None.
    spatial_key (str, optional): Key in adata.obsm for spatial coordinates. 
                                 Defaults to Key.obsm.spatial.
    copy (bool, optional): Whether to return a copy of the data or modify in place. 
                           Defaults to None.
    adj (sp.spmatrix | None, optional): Precomputed adjacency matrix. 
                                        Defaults to None.

    Returns:
    If `copy` is True, returns a tuple (zscore, count) where:
        zscore (np.ndarray): Computed z-scores for neighborhood enrichment.
        count (np.ndarray): Counts of neighbors for each cell type.
    Otherwise, the function modifies the input `adata` object by adding the 
    computed data to `adata.uns`.

    The function first checks the type of `adata`, processes the clustering and
    spatial keys, computes an adjacency matrix if not provided, and then calculates
    the neighborhood enrichment z-score.
    """

    if isinstance(adata, SpatialData):
        adata = adata.table
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_spatial_basis(adata, spatial_key)

    if adj is None:
        _assert_connectivity_key(adata, connectivity_key)
        # I prefer the transpose of the adjacency matrix. 
        # For a KNN graph, each row now sums to K.
        adj = adata.obsp[connectivity_key].T

    
    original_clust = adata.obs[cluster_key].to_numpy()
    # Set up dictionaries for converting labels to numeric labels
    unique_clust = adata.obs[cluster_key].cat.categories.values
    clust2ind = {clust : i for i,clust in enumerate(unique_clust)}

    # Create label matrix y (num points x num_labels)
    n, m = len(original_clust), len(unique_clust)
    rows = np.arange(n)
    cols = np.array([clust2ind[c] for c in original_clust])
    vals = np.ones(n, dtype='uint32')
    y = sp.csr_matrix((vals, (rows, cols)), shape=(n,m))

    start = logg.info("Running neighborhood enrichment.")
    zscore, count = _nhood_enrichment(adj, y)

    if copy:
        return zscore, count

    _save_data(
        adata,
        attr="uns",
        key=Key.uns.nhood_enrichment(cluster_key),
        data={"zscore": zscore, "count": count},
        time=start,
    )
