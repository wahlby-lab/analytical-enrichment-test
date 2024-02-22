from typing import Tuple
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

from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def _connectivity_matrix(
    xy: np.ndarray,
    method="knn",
    k: int = 5,
    r: float | None = None,
    include_self: bool = False,
) -> sp.spmatrix:
    """
    Compute the connectivity matrix of a dataset based on either k-NN or radius search.

    Parameters
    ----------
    xy : np.ndarray
        The input dataset, where each row is a sample point.
    method : str, optional (default='knn')
        The method to use for computing the connectivity.
        Can be either 'knn' for k-nearest-neighbors or 'radius' for radius search.
    k : int, optional (default=5)
        The number of nearest neighbors to use when method='knn'.
    r : float, optional (default=None)
        The radius to use when method='radius'.
    include_self : bool, optional (default=False)
        If the matrix should contain self connectivities.

    Returns
    -------
    A : sp.spmatrix
        The connectivity matrix, with ones in the positions where two points are
            connected.
    """
    if method == "knn":
        A = kneighbors_graph(xy, k, include_self=include_self).astype('bool')
    else:
        A = radius_neighbors_graph(xy, r, include_self=include_self).astype('bool')
    return A




def _nhood_enrichment(adj : sp.spmatrix, y : sp.spmatrix, regions: sp.spmatrix | None = None) -> np.ndarray:
    """
    Compute the neighborhood enrichment z-score.

    This function calculates the z-score for neighborhood enrichment, 
    a metric to assess the interaction between different types of points (e.g., cells)
    in spatial omics data based on their spatial adjacency and labels.

    Parameters:
    adj (sp.spmatrix): Interaction matrix representing the spatial interaction between points. 
                       Each entry adj[i, j] is non-zero if point i is adjacent to point j.
    y (sp.spmatrix): Binary matrix representing the labels for each point. 
                     Each row corresponds to a one-hot-encoded label of each point.

    Returns:
    np.ndarray: A tuple containing:
                - zscore: A matrix of z-scores, representing the strength of enrichment 
                          between each pair of labels.
                - total_neighbors_count: A matrix representing the total count of neighbors 
                                         for each cell type.

    The function computes the z-score by comparing the actual count of neighbors with 
    specific labels to the expected count under a random distribution of labels.
    """

    # Number of points (nodes)
    n = adj.shape[0]
    if regions is not None:
        normalize = regions.sum(axis=0)


    # Calculate the number of label A neighbors each point has
    neighbors_count = adj @ y

    # Compute mean and variance
    if regions is None:
        # Total count of neighbor interactions
        total_neighbors_count = y.T @ neighbors_count
        total_neighbors_count = total_neighbors_count.A  # Convert to dense array
        # If we randomly pick a point, what is the expected number
        # of label A neighbirs
        mean_interaction = neighbors_count.sum(axis=0).A / n
        neighbors_count.data **= 2
        var_interaction = neighbors_count.sum(axis=0).A / n - mean_interaction**2
        
        total_label_count = y.sum(axis=0).A.T
        zscore = (total_neighbors_count - total_label_count.T * mean_interaction.T) / np.sqrt(total_label_count.T * var_interaction.T)

    else:
        # Total count of neighbor interactions
        total_neighbors_count = y.T @ neighbors_count
        total_neighbors_count = total_neighbors_count.A  # Convert to dense array

        # If we randomly pick a point, what is the expected number of label A neighbors
        # within specified reigon.
        mean_interaction = regions.multiply(neighbors_count).sum(axis=0) / normalize
        mean_interaction = mean_interaction.A
        neighbors_count.data **= 2
        var_interaction = regions.multiply(neighbors_count).sum(axis=0).A / normalize - mean_interaction**2
        total_label_count = regions.T.dot(y).A.T
        zscore = (total_neighbors_count - total_label_count.T * mean_interaction.T) / np.sqrt(total_label_count.T * var_interaction.A.T)

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
    adj: sp.spmatrix | None = None,
    knn_test : Tuple[int,int] | None = None,
    radius_test : Tuple[float,float] | None = None
):
    """
    Computes the neighborhood enrichment z-scores.

    This function calculates the z-score for neighborhood enrichment, 
    a score to assess the interaction between spatial points with different categorical
    labels.

    Given three sets of points: Points with label `A`, points with label `B`
    and points with label `C`. Two points are said to interact if they are connected
    by an edge in the connectivity graph. 
    
    The enrichment score, `Z(A,B)`, indicating if points with
    label `B` are enriched around points with label `A` is given by:

        `Z(A,B) = (x - u) / s`

    where 
        `x` is the observed number of interactions between points with label `A`
            and points with label `B`
        `u` is the average number of interactions when points with label `B` are 
            randomly placed  in the connectivity graph.
        `s` is the standard deviation in number of -||-
            

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
    if cluster_key is not None:
        _assert_categorical_obs(adata, cluster_key)
    _assert_spatial_basis(adata, spatial_key)

    xy = adata.obsm[spatial_key]

    original_clust = adata.obs[cluster_key].to_numpy()
    # Set up dictionaries for converting labels to numeric labels
    unique_clust = adata.obs[cluster_key].cat.categories.values
    clust2ind = {clust : i for i,clust in enumerate(unique_clust)}
    n, m = len(original_clust), len(unique_clust)
    rows = np.arange(n)
    cols = np.array([clust2ind[c] for c in original_clust])
    vals = np.ones(n, dtype='uint32')
    y = sp.csr_matrix((vals, (rows, cols)), shape=(n,m))


    mask = None
    if knn_test is not None:
        adj = _connectivity_matrix(xy, method='knn', k=knn_test[0], include_self=False)
        adj_null = _connectivity_matrix(xy, method='knn', k=knn_test[1], include_self=False)
        mask = y.T.astype('bool').dot(adj_null).T
        mask.data = np.ones_like(mask.data)

    elif radius_test is not None:
        adj = _connectivity_matrix(xy, method='radius', r=radius_test[0], include_self=False)
        adj_null = _connectivity_matrix(xy, method='radius', r=radius_test[1], include_self=False)
        mask = y.T.astype('bool').dot(adj_null).T

    elif adj is None:
        _assert_connectivity_key(adata, connectivity_key)
        adj = adata.obsp[connectivity_key]
        _assert_connectivity_key(adata, connectivity_key)
        # I prefer the transpose of the adjacency matrix. 
        # For a KNN graph, each row now sums to K.
        adj = adj.T

    else:
        # I prefer the transpose of the adjacency matrix. 
        # For a KNN graph, each row now sums to K.
        adj = adj.T
        
    start = logg.info("Running neighborhood enrichment.")
    #zscore, count = _nhood_enrichment(adj, y)
    #zscore2, count2 = _nhood_enrichment2(adj, y)
    #mask = sp.csr_matrix(1-y.A)
    zscore, count = _nhood_enrichment(adj, y, regions=mask)
    
    if copy:
        return zscore, count

    _save_data(
        adata,
        attr="uns",
        key=Key.uns.nhood_enrichment(cluster_key),
        data={"zscore": zscore, "count": count},
        time=start,
    )




def nhood_binominal_enrichment(
    adata : AnnData | SpatialData,
    cluster_key : str,
    connectivity_key: str | None = None,
    spatial_key: str = Key.obsm.spatial,
    copy: bool = None,
    adj: sp.spmatrix | None = None
):
    
    if isinstance(adata, SpatialData):
        adata = adata.table
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    if cluster_key is not None:
        _assert_categorical_obs(adata, cluster_key)
    _assert_spatial_basis(adata, spatial_key)


    original_clust = adata.obs[cluster_key].to_numpy()

    # Set up dictionaries for converting labels to numeric labels
    unique_clust = adata.obs[cluster_key].cat.categories.values
    clust2ind = {clust : i for i,clust in enumerate(unique_clust)}
    n, m = len(original_clust), len(unique_clust)
    rows = np.arange(n)
    cols = np.array([clust2ind[c] for c in original_clust])
    vals = np.ones(n, dtype='float32')

    # Label matrix
    y = sp.csr_matrix((vals, (rows, cols)), shape=(n,m))



    _assert_connectivity_key(adata, connectivity_key)
    adj = adata.obsp[connectivity_key]
    # I prefer the transpose of the adjacency matrix. 
    # For a KNN graph, each row now sums to K.
    adj = adj.T


    # Get observed counts
    counts = y.T @ adj @ y
    counts = counts.A

    # Get number of reference cells neighboring query cells
    y1 = y.T @ adj @ y

    # Get number of neighbors for each reference cell type
    n = y1.sum(axis=1)

    # Compute proportions
    p = y1 / n
    p = p.A
    # Repeat but with randomized labels
    n = n.A
    y_rand = y.sum(axis=0)
    y_rand = y_rand / y_rand.sum()
    y_rand = y_rand 
    p_null = y_rand.T * y_rand * adj.sum() 
    p_null = p_null / p_null.sum(axis=1)
    p_null = p_null.A
    zscore = 2 * np.arcsin(np.sqrt(p)) - 2 * np.arcsin(np.sqrt(p_null)) 

    start = logg.info("Running neighborhood enrichment.")

    if copy:
        return zscore, y1

    _save_data(
        adata,
        attr="uns",
        key=Key.uns.nhood_enrichment(cluster_key),
        data={"zscore": zscore, "count": y1},
        time=start,
    )







if __name__ == '__main__':
    import numpy as np
    import anndata
    import pandas as pd
    class MakeData:
        def __init__(self):
            pass

        def _noise(self, n, step):
            return step * (np.random.rand(n, 2) - 5)
        def _get_points(self, roi, step, label):
            x = np.arange(roi[0],roi[0]+roi[2],step)
            y = np.arange(roi[1],roi[1]+roi[3],step)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.flatten(), Y.flatten()))
            #points = points + self._noise(len(points), step)
            high = roi[0] + roi[2]
            low = roi[0]
            points = np.random.rand(len(x)**2, 2) * (high-low) + low
            labels = np.array(len(points)*[label])
            return points, labels

        def dataset(self, w=3):
            np.random.seed(42)
            w0 = 0.25
            w1 = 0.25
            w2 = w
            dataset = {
                'red' : [-w0/2,-w0/2,w0,w0],
                'green' : [-w1/2,-w1/2,w1,w1],
                'blue' : [-w2/2,-w2/2,w2,w2],
                'yellow' : [-w2/2,-w2/2,w2,w2]
            } 
            results = [self._get_points(roi=roi, step=0.05, label=label) for label, roi in dataset.items()]
            points, labels = zip(*results)
            points = np.vstack(points)
            labels = np.hstack(labels)
            adata = anndata.AnnData(obsm={'spatial': points}, obs={'labels' : labels.tolist()})
            adata.obs['labels'] = adata.obs['labels'].astype('category') 
            return adata
    import matplotlib.pyplot as plt
    adata = MakeData().dataset(w=2)
    for label in adata.obs.labels.cat.categories:
        ind = adata.obs.labels == label
        plt.scatter(adata.obsm['spatial'][ind,0], adata.obsm['spatial'][ind,1], color=label, alpha=1 if label == 'gray' else 1.0, s=3)


    import scanpy as sc 
    import squidpy as sq
    from nhood_enrichment import nhood_enrichment
    import seaborn as sns

    ws = [0.25, 1, 2, 5, 10]
    z_axel = []
    z_squidpy = []
    for w in ws:
        adata = MakeData().dataset(w)

        colors =adata.obs.labels.cat.categories

        sq.gr.spatial_neighbors(adata, set_diag=False, n_neighs=10)
        zscore_squidpy, _ = sq.gr.nhood_enrichment(adata, cluster_key='labels', copy=True, n_perms=500, show_progress_bar=False)
        zscore_axel, count_axel = nhood_binominal_enrichment(adata, cluster_key='labels', copy=True)
        z_axel.append(zscore_axel[1,2])
        z_squidpy.append(zscore_squidpy[1,2])

    plt.figure()
    plt.scatter(ws,z_axel, label='axel')
    #plt.scatter(ws,z_squidpy, label='squidpy')
    plt.show()    
    g = sns.clustermap(zscore_squidpy, annot=True, row_colors=colors, col_colors=colors, row_cluster=False, col_cluster=False, vmin=-np.abs(zscore_squidpy).max(), vmax=np.abs(zscore_squidpy).max(), cmap='seismic')
    g.ax_heatmap.set_title('Squidpy enrichment', pad=30, fontsize=30, fontweight='bold')
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_yticks([])

    g = sns.clustermap(zscore_axel, row_colors=colors, col_colors=colors, row_cluster=False, col_cluster=False, annot=True, vmin=-np.abs(zscore_axel).max(), vmax=np.abs(zscore_axel).max(), cmap='seismic')
    g.ax_heatmap.set_title('Analytical enrichment',pad=30, fontsize=30, fontweight='bold')
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_yticks([])

    plt.show()
