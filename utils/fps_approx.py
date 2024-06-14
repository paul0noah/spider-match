import scipy.sparse as sparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import igl

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

def fps_wrapper(v, f, n=2):
    _, vv, ff, _, iv = igl.decimate(v, f, 2*max(n, 20000))
    D = geodesic_distmat_dijkstra(vv, ff)
    points = farthest_point_sampling_distmat(D, n, random_init=False, verbose=False)

    return iv[points]

import networkx as nx
import igl
def metric_salesman_approx(v, f, augmented_edges=False):
    e = igl.edges(f)
    e = np.append(e, e[:, [1, 0]], axis=0)
    if augmented_edges:
        from utils.sm_utils import get_augmented_edge
        e = get_augmented_edge(v, f)
    elen = np.linalg.norm(v[e[:, 1], :] - v[e[:, 0], :], axis=1)
    G = nx.Graph()
    i = 0
    for edge in e:
        G.add_edge(edge[0], edge[1], weight=elen[i])
        i = i + 1

    # solve salesman approx
    T = nx.minimum_spanning_tree(G, weight='weight')

    dfs = nx.dfs_preorder_nodes(T, 1)
    listnode = []
    for item in dfs:
        listnode += [item]

    cycle = nx.approximation.traveling_salesman_problem(G, weight="weight", nodes=G.nodes(), cycle=True)
    return cycle

def metric_salesman_approx_line_graph(v, f):
    if f.shape[1] == 3:
        e = igl.edges(f)
        #e = np.append(e, e[:, [1, 0]], axis=0)
    else:
        e = f
    elen = np.linalg.norm(v[e[:, 1], :] - v[e[:, 0], :], axis=1)
    G = nx.Graph()
    i = 0
    for edge in e:
        G.add_edge(edge[0], edge[1], weight=elen[i])
        i = i + 1

    H = nx.line_graph(G)
    for edge in H.edges():
        elen = G[edge[0][0]][edge[0][1]]['weight'] + G[edge[1][0]][edge[1][1]]['weight']
        H.add_edge(edge[0], edge[1], weight=1)

    # solve salesman approx
    T = nx.minimum_spanning_tree(H, weight='weight')

    dfs = nx.dfs_preorder_nodes(T, edge[0])
    listnode = []
    for item in dfs:
        listnode += [item]

    cycle = nx.approximation.traveling_salesman_problem(H, weight="weight", nodes=H.nodes(), cycle=True)
    #init_cycle = None
    #for cycle in nx.simple_cycles(H):
    #    if len(cycle) == H.number_of_nodes():
    #        init_cycle = cycle
    #        break
    #cycle = nx.approximation.simulated_annealing_tsp(H, cycle)
    true_cycle = np.zeros((2 * len(cycle), 2), dtype=int)
    i = 0
    previous_idx = cycle[0][0] if cycle[0][1] == cycle[1][0] else cycle[0][1]
    cycle.append(cycle[0])
    for j in range(len(cycle)-1):
        edge = cycle[j]
        if edge[0] == previous_idx:
            true_cycle[i, 0] = edge[0]
            true_cycle[i, 1] = edge[1]
        else:
            true_cycle[i, 1] = edge[0]
            true_cycle[i, 0] = edge[1]
        previous_idx = true_cycle[i, 1]
        i = i + 1
        # we take the edge twice
        if cycle[j+1][0] != previous_idx and cycle[j+1][1] != previous_idx:
            true_cycle[i, 0] = true_cycle[i-1, 1]
            true_cycle[i, 1] = true_cycle[i-1, 0]
            previous_idx = true_cycle[i, 1]
            i = i + 1
    return true_cycle[:i]

def combine_cycle_and_paths(e, pair_paths, cycle):
    edge_to_idx = {}
    for i in range(e.shape[0]):
        edge_to_idx[tuple(e[i, :].tolist())] = i
        edge_to_idx[tuple(e[i, [1, 0]].tolist())] = i

    path_indices = []
    for j in range(cycle.shape[0]-1):
        key = tuple(cycle[j, :].tolist())
        idx = edge_to_idx[key]
        path = pair_paths[idx]
        src = cycle[j, 0]
        trgt = cycle[j, 1]
        if path[0, 0] != src:
            path = np.flip(path)
        if path[-1, 1] != trgt or path[0, 0] != src:
            #print(path)
            print(src)
            print(trgt)
            print("Big error")
            raise Exception
        path_indices = path_indices + path[:, 0].tolist() + [trgt]

    return path_indices

def m_tsp_ant_colony_line_graph(v, f):
    #doesnt work lol
    from utils.m_tsp import Solver, Colony

    if f.shape[1] == 3:
        e = igl.edges(f)
        #e = np.append(e, e[:, [1, 0]], axis=0)
    else:
        e = f
    elen = np.linalg.norm(v[e[:, 1], :] - v[e[:, 0], :], axis=1)
    G = nx.Graph()
    i = 0
    for edge in e:
        G.add_edge(edge[0], edge[1], weight=elen[i])
        i = i + 1

    H = nx.line_graph(G)
    for edge in H.edges():
        elen = G[edge[0][0]][edge[0][1]]['weight'] + G[edge[1][0]][edge[1][1]]['weight']
        H.add_edge(edge[0], edge[1], weight=1)

    HH = nx.Graph()
    node_to_idx = {}
    idx_to_node = {}
    i = 0
    for node in H.nodes():
        HH.add_node(i)
        node_to_idx[node] = i
        idx_to_node[i] = node
        i = i + 1
    for edge in H.edges():
        HH.add_edge(node_to_idx[edge[0]], node_to_idx[edge[1]], weight=1)


    solver = Solver()
    colony = Colony(alpha=1, beta=3) # alpha
    # num of sales
    sales = 5
    start_node = 0
    max_cycles = 15
    num_try_opt2 = 0
    ans = solver.solve(HH, colony, sales, start=start_node, limit=max_cycles, opt2=num_try_opt2)

def find_cycles(v, f, min_cycle_len=4, max_cycle_len=6):
    if f.shape[1] == 3:
        e = igl.edges(f)
        # e = np.append(e, e[:, [1, 0]], axis=0)
    else:
        e = f
    G = nx.Graph()
    i = 0
    for edge in e:
        G.add_edge(edge[0], edge[1], weight=1)
        i = i + 1
    H = nx.line_graph(G)
    for edge in H.edges():
        elen = G[edge[0][0]][edge[0][1]]['weight'] + G[edge[1][0]][edge[1][1]]['weight']
        H.add_edge(edge[0], edge[1], weight=elen)

    for cycle in nx.simple_cycles(H, length_bound=max_cycle_len):
        if len(cycle) == max_cycle_len:
            break

def get_angle_on_plane(edge_dir_0, edge_dir_1, normal):
    edge_dir_0 = edge_dir_0 / np.linalg.norm(edge_dir_0)
    edge_dir_1 = edge_dir_1 / np.linalg.norm(edge_dir_1)
    proj_edge_dir_0 = edge_dir_0 - np.dot(edge_dir_0, normal) * normal
    proj_edge_dir_0 = proj_edge_dir_0 / np.linalg.norm(proj_edge_dir_0)
    proj_edge_dir_1 = edge_dir_1 - np.dot(edge_dir_1, normal) * normal
    proj_edge_dir_1 = proj_edge_dir_1 / np.linalg.norm(proj_edge_dir_1)
    angle = np.arccos(np.clip(np.dot(proj_edge_dir_0, proj_edge_dir_1), -1.0, 1.0))
    return angle
def straight_cycle(v, f, pairs):
    LEN_WEIGHT = 0.5
    REPELLING_FORCE_WEIGHT = 0
    sqrt_area = np.sqrt(igl.doublearea(v, f).sum())
    if REPELLING_FORCE_WEIGHT > 0:
        d = geodesic_distmat_dijkstra(v, f)
        d = d / sqrt_area
        max_d = d.max()

    #print("modifying point_idxs")
    #point_idxs = point_idxs[:2]
    pair_paths = []
    pair_points, pair_points_count = np.unique(pairs[:], return_counts=True)
    success = np.zeros_like(pairs[:, 0], dtype=bool)

    # build line graph from edges of x
    normals = igl.per_vertex_normals(v, f)
    e = igl.edges(f)
    e = np.append(e, e[:, [1, 0]], axis=0)
    elen_dict = {}
    for edge in e:
        elen_dict[tuple(edge.tolist())] = np.linalg.norm(v[edge[1], :] - v[edge[0], :]) / sqrt_area

    G = nx.DiGraph()
    i = 0
    for edge in e:
        G.add_edge(edge[0], edge[1])
        i = i + 1

    # cost is straight line
    H = nx.line_graph(G)
    for edge in H.edges():
        normal_idx = edge[0][1]
        normal = normals[normal_idx, :]
        edge_dir_0 = v[edge[0][1], :] - v[edge[0][0], :]
        edge_dir_1 = v[edge[1][1], :] - v[edge[1][0], :]
        angle = get_angle_on_plane(edge_dir_0, edge_dir_1, normal)

        elen = angle ** 2 + LEN_WEIGHT * elen_dict[edge[0]]
        H.add_edge(edge[0], edge[1], weight=elen)


    # GUROBI OPTIMISATION

    npoints = pairs.shape[0]
    sol_edges = np.zeros_like(e)
    added_sol = 0
    for oi in range(npoints):
        try:
            if REPELLING_FORCE_WEIGHT > 0 and added_sol > 0:
                points_on_path = np.unique(sol_edges[:added_sol, :])
                dmin = d[:, points_on_path].min(axis=1)
                dmin = max_d - dmin

            m = gp.Model()
            x = gp.tupledict()
            for (v1, v2) in H.edges:
                obj = H[v1][v2]['weight']
                if REPELLING_FORCE_WEIGHT > 0 and added_sol > 0:
                    obj = obj + REPELLING_FORCE_WEIGHT * 0.5 * (dmin[v1[0]] + dmin[v2[0]])
                x[v1, v2] = m.addVar(vtype="B", obj=obj, name=f"x_{v1}_{v2}")


            src  = pairs[oi, 0]
            trgt = pairs[oi, 1]
            # adding flow conservation constraints
            for node in H.nodes:
                if node[0] == src or node[0] == trgt:
                    continue

                # we collect the predecessor variables
                expr1 = gp.quicksum(x[i, node] for i in H.predecessors(node))

                # we collect the successor variables
                expr2 = gp.quicksum(x[node, j] for j in H.successors(node))

                # we add the constraint
                m.addLConstr(expr1 - expr2 == 0)

            # the extracted local path should not cross itself ie every vertex should just be used once
            for current_node in G.nodes:
                expr = None
                for prev_node in G.predecessors(current_node):
                    node = (prev_node, current_node)
                    if expr is None:
                        expr = gp.quicksum(x[i, node] for i in H.predecessors(node))
                    else:
                        expr = expr + gp.quicksum(x[i, node] for i in H.predecessors(node))
                m.addLConstr(expr <= 1)

            # src has to be matched
            for point in [src, trgt]:
                edges_of_point = e[np.logical_or(e[:, 0] == point, e[:, 1] == point), :]
                expr = None
                for edge in edges_of_point:
                    edge_tuple = tuple(edge.tolist())
                    if expr is None:
                        expr = gp.quicksum(x[edge_tuple, i] for i in H.successors(edge_tuple))
                        # expr = expr + gp.quicksum(x[i, edge_tuple] for i in H.predecessors(edge_tuple))
                    else:
                        expr = expr + gp.quicksum(x[edge_tuple, i] for i in H.successors(edge_tuple))
                        # expr = expr + gp.quicksum(x[i, edge_tuple] for i in H.predecessors(edge_tuple))
                m.addLConstr(expr == 1)

            # for vertices with only two adjacent edges we want to enforces straight follow up edges
            if True:
                src_idx = np.argwhere(src == pair_points)[0, 0]
                s_edges = sol_edges[:added_sol, :]
                if pair_points_count[src_idx] == 2:
                    edg = s_edges[np.logical_or(s_edges[:, 0] == src, s_edges[:, 1] == src)].flatten()
                    if edg.size != 0:
                        if src == edg[0]:
                            edg = edg[[1, 0]]
                        edg = tuple(edg.tolist())
                        edge_dir_0 = v[edg[1], :] - v[edg[0], :]
                        normal = normals[src]
                        min_angle = 10000
                        min_edge = None
                        for edg2 in H.successors(edg):
                            edge_dir_1 = v[edg2[1], :] - v[edg2[0], :]
                            angle = get_angle_on_plane(edge_dir_0, edge_dir_1, normal)
                            if angle < min_angle:
                                min_angle = angle
                                min_edge = edg2
                        expr = None
                        for edg2 in H.successors(edg):
                            if edg2 != min_edge:
                                if expr is None:
                                    expr = x[edg, edg2]
                                else:
                                    expr = expr + x[edg, edg2]
                        m.addConstr(expr == 0)
                trgt_idx = np.argwhere(trgt == pair_points)[0, 0]
                if pair_points_count[trgt_idx] == 2:
                    edg = s_edges[np.logical_or(s_edges[:, 0] == trgt, s_edges[:, 1] == trgt)].flatten()
                    if edg.size != 0:
                        if trgt == edg[0]:
                            edg = edg[[1, 0]]
                        edg = tuple(edg.tolist())
                        edge_dir_0 = v[edg[1], :] - v[edg[0], :]
                        normal = normals[trgt]
                        min_angle = 10000
                        min_edge = None
                        for edg2 in H.successors(edg):
                            edge_dir_1 = v[edg2[1], :] - v[edg2[0], :]
                            angle = get_angle_on_plane(edge_dir_0, edge_dir_1, normal)
                            if angle < min_angle:
                                min_angle = angle
                                min_edge = edg2
                        expr = None
                        for edg2 in H.successors(edg):
                            if edg2 != min_edge:
                                if expr is None:
                                    expr = x[edg, edg2]
                                else:
                                    expr = expr + x[edg, edg2]
                        m.addConstr(expr == 0)
            '''
            # every point should be matched once
            for point in point_idxs:
                edges_of_point = e[np.logical_or(e[:, 0] == point, e[:, 1] == point), :]
                expr = None
                for edge in edges_of_point:
                    edge_tuple = tuple(edge.tolist())
                    if expr is None:
                        expr = gp.quicksum(x[edge_tuple, i] for i in H.successors(edge_tuple))
                        #expr = expr + gp.quicksum(x[i, edge_tuple] for i in H.predecessors(edge_tuple))
                    else:
                        expr = expr + gp.quicksum(x[edge_tuple, i] for i in H.successors(edge_tuple))
                        #expr = expr + gp.quicksum(x[i, edge_tuple] for i in H.predecessors(edge_tuple))
                m.addLConstr(expr == 1)
            '''
            # we dont want to use already used edges and their inverse orientation
            for ee in range(added_sol):
                node = tuple(sol_edges[ee].tolist())
                expr1 = gp.quicksum(x[i, node] for i in H.predecessors(node))
                expr2 = gp.quicksum(x[node, j] for j in H.successors(node))
                m.addLConstr(expr1 + expr2 == 0)

                node = tuple(sol_edges[ee, [1, 0]].tolist())
                expr1 = gp.quicksum(x[i, node] for i in H.predecessors(node))
                expr2 = gp.quicksum(x[node, j] for j in H.successors(node))
                m.addLConstr(expr1 + expr2 == 0)

            # we dont want to go through points in pairs if they are not current src or trgt to avoid infeasibilities with
            # previous constraint
            for p in pair_points:
                if p in [src, trgt]:
                    continue
                expr = None
                for prev_node in G.predecessors(p):
                    node = (prev_node, p)
                    if expr is None:
                        expr = gp.quicksum(x[i, node] for i in H.predecessors(node))
                    else:
                        expr = expr + gp.quicksum(x[i, node] for i in H.predecessors(node))
                m.addLConstr(expr == 0)

            m.optimize()
            if m.status == 3:
                print("Model Infeasible")
                print("++++++++++++++++++++++++++++++++")
                print(f"\n\n\nCould not find path from {src} to {trgt}\n\n\n")
                print("++++++++++++++++++++++++++++++++")
                continue
                break

            local_sol_edges = np.zeros_like(e)
            local_num_added = 0
            for xx in x:
                temp = x.get(xx)
                if temp.X == 1:
                    local_sol_edges[local_num_added, :] = np.array(xx[0])
                    local_num_added = local_num_added + 1
            local_sol_edges = np.unique(local_sol_edges[:local_num_added], axis=0)
            n_local = local_sol_edges.shape[0]
            sorted_edges = -np.ones((n_local, 2), dtype=int)
            flip_src_trgt = False
            try:
                idx = np.argwhere(local_sol_edges[:, 0] == src).flatten()[0]
            except:
                idx = np.argwhere(local_sol_edges[:, 0] == trgt).flatten()[0]
                flip_src_trgt = True
            for i in range(n_local):
                try:
                    sorted_edges[i] = local_sol_edges[idx]
                    next_vert_idx = sorted_edges[i, 1]
                    if next_vert_idx == trgt and not flip_src_trgt:
                        break
                    if next_vert_idx == src and flip_src_trgt:
                        break

                    idx = np.argwhere(local_sol_edges[:, 0] == next_vert_idx).flatten()[0]
                except:
                    print(i)
            if (sorted_edges == -1).any():
                print("Failed")
                continue
            if sorted_edges[0, 0] != src or sorted_edges[-1, 1] != trgt:
                if sorted_edges[0, 0] != trgt or sorted_edges[-1, 1] != src:
                    print("Failed")
                    continue
            sol_edges[added_sol:added_sol+n_local] = sorted_edges
            added_sol = added_sol + n_local
            pair_paths.append(sorted_edges)
            success[oi] = True
            print("done")
        except Exception as e:
            print(e)

    sol_edges = sol_edges[:added_sol, :]
    return sol_edges, pair_paths, success
    import polyscope as ps
    ps.init()
    ps.register_surface_mesh('m', v, f)
    ps.register_point_cloud('fps', v[np.unique(pairs[:]), :], radius=0.01)
    ps.register_point_cloud('sol_points', v[sol_edges[:, 0], :], radius=0.01)

    for pair_path in pair_paths:
        npoints = pair_path.shape[0]
        edgs = np.column_stack((np.arange(0, npoints), np.arange(1, npoints+1)))
        verts = np.zeros((npoints+1, 3))
        verts[:-1] = v[pair_path[:, 0], :]
        verts[-1] = v[pair_path[-1, 1]]
        ps.register_curve_network("c" + str(pair_path[0, 0]) + "-" + str(pair_path[-1, 1]), verts, edgs, material='candy')
    #npoints = sol_edges.shape[0]
    #ps.register_curve_network('c', v[sol_edges[:, 0], :], np.column_stack((np.arange(0, npoints - 1), np.arange(1, npoints))))
    ps.show()

def get_geodist_edges(v, f, points, upper_bound_edges=4):
    d = geodesic_distmat_dijkstra(v, f)
    d = d / d.max()
    d = d[points, :]
    d = d[:, points]
    npoints = 3
    npointsi = (d < 0.5).astype('int').sum(axis=1)
    #npointsi[npointsi > 4] = 4
    adj_list = igl.adjacency_list(f)
    for i in range(points.shape[0]):
        if npointsi[i] > len(adj_list[points[i]])-1:
            npointsi[i] = len(adj_list[points[i]])-1
        if npointsi[i] > upper_bound_edges:
            npointsi[i] = upper_bound_edges
        if npointsi[i] == 3: # round to even number
            npointsi[i] = 2
        if npointsi[i] < 2:
            npointsi[i] = 2
    smol_to_large = np.lexsort((npointsi, -d.sum(axis=0)))
    d = d[smol_to_large, :]
    d = d[:, smol_to_large]
    npointsi = npointsi[smol_to_large]
    points = points[smol_to_large]
    smol_idx = np.argsort(d, axis=1)#[:, :npoints+1]
    edges = np.zeros((npoints * points.shape[0], 2), dtype=int)
    edgelens = np.zeros((npoints * points.shape[0], 1))
    num_added = 0
    p_counter = np.zeros_like(npointsi)
    points_to_idx = {}
    for i in range(points.shape[0]):
        points_to_idx[points[i]] = i
        for j in range(points.shape[0]-1):
            if p_counter[i] >= npointsi[i]:
                break
            if p_counter[smol_idx[i, j+1]] >= npointsi[smol_idx[i, j+1]]:
                continue
            src = points[i]
            trgt = points[smol_idx[i, j+1]]
            if trgt in points[:i]:
                continue #make sure we dont add extra edges for nodes which should recieve less
            new_edge = np.array([src, trgt])
            if not (np.abs(edges - new_edge[[1, 0]]).sum(axis=1) == 0).any():
                # only add new edge if inverse direction does not exists
                edges[num_added, :] = new_edge
                edgelens[num_added] = d[i, smol_idx[i, j+1]]
                num_added = num_added + 1
                p_counter[i] = p_counter[i] + 1
                p_counter[smol_idx[i, j+1]] = p_counter[smol_idx[i, j+1]] + 1
    if (p_counter < 2).any():
        print("--------------- > Point with less than 2 connections extracted")
    G = nx.from_edgelist(edges[:num_added, :])
    n_components = nx.number_connected_components(G)
    if n_components > 1:
        for _ in range(n_components-1):
            G = nx.from_edgelist(edges[:num_added, :])
            components = [np.array(list(component)) for component in nx.connected_components(G)]
            comp0idx = [points_to_idx[i] for i in components[0]]
            comp1idx = [points_to_idx[i] for i in components[1]]
            min_dist = d.max()
            new_edge = [0, 0]
            for c0idx in comp0idx:
                for c1idx in comp1idx:
                    if d[c0idx, c1idx] < min_dist:
                        if p_counter[c0idx] < len(adj_list[points[c0idx]])-1 and p_counter[c1idx] < len(adj_list[points[c1idx]])-1:
                            min_dist = d[c0idx, c1idx]
                            new_edge = [points[c0idx], points[c1idx]]

            if new_edge[0] == 0 and new_edge[1] == 0:
                print("Failed to recover from 2 disconnected components")
                raise RuntimeError
            edges[num_added, :] = np.array(new_edge)
            edgelens[num_added] = min_dist
            num_added = num_added + 1
    G = nx.from_edgelist(edges[:num_added, :])
    if nx.number_connected_components(G) > 1:
        print("Could not recover from disconnected components")
        raise RuntimeError
    edges = edges[:num_added, :]
    edgelens = edgelens[:num_added]
    edges = edges[np.argsort(edgelens.flatten())]
    return edges
def salesman(v, f):
    e = igl.edges(f)
    e = np.append(e, e[:, [1, 0]], axis=0)
    elen = np.linalg.norm(v[e[:, 1], :] - v[e[:, 0], :], axis=1)

    m = gp.Model()

    xe = m.addMVar(shape=e.shape[0], vtype=GRB.BINARY, name="x")
    for i in range(v.shape[0]):
        proj0 = e[:, 0] == i
        proj1 = e[:, 1] == i
        m.addConstr(proj0 @ xe == 1, name="vertexout" + str(i))
        m.addConstr(proj1 @ xe == 1, name="vertexin" + str(i))

    m.setObjective(xe.sum(), GRB.MINIMIZE)
    m.optimize()

    tour = e[xe.X.astype('bool')]

    cleanedtour = np.zeros_like(tour)
    cleanedtour[0, :] = tour[0, :]
    edgeused = np.zeros(shape=(tour.shape[0], 1), dtype='bool')
    for i in range(1, tour.shape[0]-1):
        next_src = cleanedtour[i-1, 1]

        potential_edges = np.where(tour[:, 0] == next_src)
        for ii in potential_edges:
            if not edgeused[ii] and tour[ii, 0] == next_src:
                edgeused[ii] = True
                cleanedtour[i, :] = tour[ii, :]
                break

    return cleanedtour[:, 0]