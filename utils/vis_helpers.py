import scipy.sparse as sparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

def farthest_point_sampling_distmat(D, k, random_init=True, verbose=False):
    """
    Samples points using farthest point sampling using a complete distance matrix
    Parameters
    -------------------------
    D           : (n,n) distance matrix between points
    k           : int - number of points to sample
    random_init : Whether to sample the first point randomly or to
                  take the furthest away from all the other ones
    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """
    if random_init:
        rng = np.random.default_rng()
        inds = [rng.integers(D.shape[0]).item()]
    else:
        inds = [np.argmax(D.sum(1))]

    dists = D[inds[0]]

    iterable = range(k-1) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k-1:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, D[newid])

    return np.asarray(inds)

def edges_from_faces(faces):
    """
    Compute all edges in the mesh
    Parameters
    --------------------------------
    faces : (m,3) array defining faces with vertex indices
    Output
    --------------------------
    edges : (p,2) array of all edges defined by vertex indices
            with no particular order
    """
    # Number of vertices
    N = 1 + np.max(faces)

    # Use a sparse matrix and find non-zero elements
    # This is way faster than a np.unique somehow
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    V = np.ones_like(I)

    M = sparse.coo_matrix((V, (I, J)), shape=(N, N))

    edges0 = M.row
    edges1 = M.col

    indices = M.col > M.row
    edges = np.concatenate([edges0[indices, None], edges1[indices, None]], axis=1)
    return edges

def geodesic_distmat_dijkstra(vertices, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm.
    """
    N = vertices.shape[0]
    edges = edges_from_faces(faces)

    I = edges[:, 0]  # (p,)
    J = edges[:, 1]  # (p,)
    V = np.linalg.norm(vertices[J] - vertices[I], axis=1)  # (p,)

    graph = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsc()

    geod_dist = sparse.csgraph.shortest_path(graph, directed=False)

    return geod_dist


def knn_search(x, X, k=1):
    """
    find indices of k-nearest neighbors of x in X
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(x)
    if k == 1:
        return indices.flatten()
    else:
        return indices