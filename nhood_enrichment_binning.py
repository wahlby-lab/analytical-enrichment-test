
import numpy as np
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

def sort_labels(array):
    # Perform hierarchical clustering
    # You can choose different methods and metrics if needed
    Y = pdist(array)
    Z = linkage(Y, method='average')

    # Get the order of the leaves of the hierarchical tree
    sorted_indices = leaves_list(Z)
    return sorted_indices


def label_matrix(labels, dtype=bool, return_uniques=False):
    unique_labels, ind = np.unique(labels, return_inverse=True)
    n,m = len(labels), len(unique_labels)
    values = np.ones(n, dtype=dtype)
    y = csr_matrix((values,(np.arange(n), ind)), shape=(n,m))
    if return_uniques:
        return y, unique_labels
    else:
        return y

def spatial_binning_matrix(
    xy: np.ndarray, bin_width: float
) -> csr_matrix:


    # Compute shifted coordinates
    mi, ma = xy.min(axis=0, keepdims=True), xy.max(axis=0, keepdims=True)
    xys = xy - mi

    # Compute grid size
    grid = ma - mi
    grid = grid.flatten()

    # Compute bin index
    bin_ids = xys // bin_width
    bin_ids = bin_ids.astype("int")
    bin_ids = tuple(x for x in bin_ids.T)

    # Compute grid size in indices
    size = grid // bin_width + 1
    size = tuple(x for x in size.astype("int"))

    # Convert bin_ids to integers
    linear_ind = np.ravel_multi_index(bin_ids, size)

    # Create a matrix indicating which markers fall in what bin
    bin_matrix = label_matrix([str(i) for i in linear_ind], dtype=bool)
    bin_matrix = bin_matrix.T

    return bin_matrix


def nhood_enrichment(xy: np.ndarray, labels:np.ndarray, bin_width:float) -> np.ndarray:
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
    bin_width (float): Width of each bin used for creating the interaction graph.

    Returns:
    np.ndarray: A tuple containing:
                - zscore: A matrix of z-scores, representing the strength of enrichment 
                          between each pair of labels.
                - total_neighbors_count: A matrix representing the total count of neighbors 
                                         for each cell type.
                - unique_labels : List of unique labels matching the rows and columns of the zscore
                                        matrix.

    The function computes the z-score by comparing the actual count of neighbors with 
    specific labels to the expected count under a random distribution of labels.
    """

    n = len(labels)

    # Create matrix with one-hot-encoded labels
    y, unique_labels = label_matrix(labels, dtype='uint32', return_uniques=True)

    # Create adjacency matrix using binning
    adj = spatial_binning_matrix(xy, bin_width)
    adj = adj.T @ adj
    adj.setdiag(False)

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

    ind = sort_labels(zscore)

    # Sort labels
    zscore = zscore[ind,:]
    zscore = zscore[:,ind]
    total_neighbors_count = total_neighbors_count[ind,:]
    total_neighbors_count = total_neighbors_count[:,ind]
    unique_labels = unique_labels[ind]
    return zscore, total_neighbors_count, unique_labels


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv(r'https://tissuumaps.dckube.scilifelab.se/private/Points2Regions/toy_data.csv')

    xy = data[['X','Y']].to_numpy()
    labels = data['Genes'].to_numpy()

    zscore, count, unique_labels = nhood_enrichment(xy, labels, 100)