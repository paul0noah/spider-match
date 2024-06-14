import os
import numpy as np
import trimesh
import networkx as nx
import open3d as o3d

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def compute_geodesic_distmat(verts, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """
    NN = 500

    for NN in np.arange(500, 2000, 500):
        # get adjacency matrix
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        vertex_adjacency = mesh.vertex_adjacency_graph
        assert nx.is_connected(vertex_adjacency), 'Graph not connected'
        vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
        # get adjacency distance matrix
        graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
        distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
        distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
        # compute geodesic matrix
        geodesic_x = shortest_path(distance_adj, directed=False)
        if np.any(np.isinf(geodesic_x)):
            print('Inf number in geodesic distance. Increasing NN.')
        else:
            break
    if np.any(np.isinf(geodesic_x)):
        print('Inf number in geodesic distance. Increasing NN failed.')
    return geodesic_x


def read_shape(file, as_cloud=False):
    """
    Read mesh from file.

    Args:
        file (str): file name
        as_cloud (bool, optional): read shape as point cloud. Default False
    Returns:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3] or None
    """
    if as_cloud:
        verts = np.asarray(o3d.io.read_point_cloud(file).points)
        faces = None
    else:
        mesh = o3d.io.read_triangle_mesh(file)
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    return verts, faces


def write_off(file, verts, faces):
    with open(file, 'w') as f:
        f.write("OFF\n")
        f.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
        for x in verts:
            f.write(f"{' '.join(map(str, x))}\n")
        for x in faces:
            f.write(f"{len(x)} {' '.join(map(str, x))}\n")
