# code adapted from original DiffusionNet implementation https://github.com/nmwsharp/diffusion-net

import os
import os.path as osp
import random
import hashlib
import numpy as np

import scipy
import scipy.spatial
import scipy.sparse.linalg as sla
import sklearn.neighbors as neighbors

import robust_laplacian
import potpourri3d as pp3d

import torch


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        if arr is not None:
            binarr = arr.view(np.uint8)
            running_hash.update(binarr)
    return running_hash.hexdigest()


def torch2np(tensor):
    assert isinstance(tensor, torch.Tensor)
    return tensor.detach().cpu().numpy()


def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()


def sparse_torch_to_np(A):
    assert len(A.shape) == 2

    indices = torch2np(A.indices())
    values = torch2np(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()
    return mat


def to_basis(feat, basis, massvec):
    """
    Transform feature into coefficients of orthonormal basis.
    Args:
        feat (torch.Tensor): feature vector [B, V, C]
        basis (torch.Tensor): functional basis [B, V, K]
        massvec (torch.Tensor): mass vector [B, V]
    Returns:
        coef (torch.Tensor): coefficient of basis [B, K, C]
    """
    basis_t = basis.transpose(-2, -1)
    coef = torch.matmul(basis_t, feat * massvec.unsqueeze(-1))
    return coef


def from_basis(coef, basis):
    """
    Transform coefficients of orthonormal basis into feature.
    Args:
        coef (torch.Tensor): coefficients [B, K, C]
        basis (torch.Tensor): functional basis [B, V, K]
    Returns:
        feat (torch.Tensor): feature vector [B, V, C]
    """
    feat = torch.matmul(basis, coef)
    return feat


def dot(a, b, keepdim=False):
    """
    Compute the dot product between vector a and vector b in last dimension

    Args:
        a (torch.Tensor): vector a [N, C].
        b (torch.Tensor): vector b [N, C].
        keepdim (bool, optional): keep dimension.
    Return:
        (torch.Tensor): dot product between a and b [N] or [N, 1].
    """
    assert a.shape == b.shape
    return torch.sum(a * b, dim=-1, keepdim=keepdim)


def cross(a, b):
    """
    Compute the cross product between vector a and vector b in last dimension

    Args:
        a (torch.Tensor): vector a [N, 3].
        b (torch.Tensor): vector b [N, 3].
    Return:
        (torch.Tensor): cross product between a and b [N, 3].
    """
    assert a.shape == b.shape and a.shape[-1] == 3
    return torch.cross(a, b, dim=-1)


def norm(x, keepdim=False):
    """
    Compute norm of an array of vectors.
    Given (N, C), return (N) or (N, 1) after norm along last dimension.
    """
    return torch.norm(x, dim=-1, keepdim=keepdim)


def square_norm(x, keepdim=False):
    """
    Compute square norm of an array of vectors.
    Given (N, C), return (N) after norm along last dimension.
    """
    return dot(x, x, keepdim=keepdim)


def normalize(x, eps=1e-12):
    """
    Normalize an array of vectors along last dimension.
    Given (N, C), return (N, C) after normalization.
    """
    assert x.dim() != 1
    return x / (norm(x, keepdim=True) + eps)


def face_coords(verts, faces):
    """
    Return face coordinates.
    Args:
        verts (torch.Tensor): vertices [V, 3]
        faces (torch.LongTensor): faces [F, 3]
    Return:
        coords (torch.Tensor): face coordinates [F, 3, 3]
    """
    coords = verts[faces]
    return coords


def project_to_tangent(vecs, normals):
    """
    Compute the tangent vectors of normals by vecs - proj(vecs, normals).
    Args:
        vecs (torch.Tensor): vecs [V, 3].
        normals (torch.Tensor): normal vectors assume to be unit [V, 3].
    """
    return vecs - dot(vecs, normals, keepdim=True) * normals


def face_area(verts, faces):
    """
    Compute face areas
    Args:
        verts (torch.Tensor): verts [V, 3]
        faces (torch.LongTensor): faces [F, 3]
    """
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    # compute area by cross product
    normal = cross(vec_A, vec_B)
    return 0.5 * norm(normal)


def face_normal(verts, faces, is_normalize=True):
    """
    Compute face normal
    Args:
        verts (torch.Tensor): verts [V, 3]
        faces (torch.LongTensor): faces [F, 3]
        is_normalize (bool, optional): whether normalize face normal. Default True.
    """
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    normal = cross(vec_A, vec_B)

    if is_normalize:
        normal = normalize(normal)

    return normal


def neighborhood_normal(pts):
    """
    Compute point cloud normal by performing PCA in neighborhood points.
    Args:
        pts (np.ndarray): points [V, N, 3], N: number of neighbors.
    """
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)


def mesh_vertex_normal(verts, faces):
    """
    Compute mesh vertex normal by adding neighboring faces' normals.
    Args:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3]
    Return:
        vertex_normals (np.ndarray): vertex normals [V, 3]
    """
    face_n = torch2np(face_normal(torch.tensor(verts), torch.tensor(faces)))

    vertex_normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)

    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=-1, keepdims=True) + 1e-12)

    return vertex_normals


def vertex_normal(verts, faces, n_neighbors=30):
    """
    Compute vertex normal supported by both point cloud and mesh

    Args:
        verts (torch.Tensor): vertices [V, 3].
        faces (torch.Tensor): faces [F, 3].
        n_neighbors (int, optional): number of neighbors to compute normal for point cloud. Default 30.
    """
    verts_np = torch2np(verts)

    if faces is None: # point cloud
        _, neigh_inds = find_knn(verts, verts, n_neighbors, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[torch2np(neigh_inds), :]
        neigh_points = neigh_points - verts_np[:, None, :]
        normals = neighborhood_normal(neigh_points)
    else:
        faces_np = torch2np(faces)
        normals = mesh_vertex_normal(verts_np, faces_np)

        # if any NaN, wiggle slightly and recompute
        bad_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
            wiggle_verts = verts_np + bad_mask * wiggle
            normals = mesh_vertex_normal(wiggle_verts, faces_np)

        # if still NaN assign random normals (probably unreferenced verts in mesh)
        bad_mask = np.isnan(normals).any(axis=1)
        if bad_mask.any():
            normals[bad_mask, :] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_mask, :]
            normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)

    if torch.any(torch.isnan(normals)):
        raise ValueError('NaN normals')

    return normals


def find_knn(src_pts, target_pts, k, largest=False, omit_diagonal=False, method='brute'):
    """
    Finds the k nearest neighbors of source on target
    Args:
        src_pts (torch.Tensor): source points [Vs, 3]
        target_pts (torch.Tensor): target points [Vt, 3]
        k (int): number of neighbors
        largest (bool, optional): whether k largest neighbors. Default False.
        omit_diagonal (bool, optional): whether omit the point itself. Default False.
        method (str, optional): method, support 'brute', 'cpu_kd'. Default 'brute'
    Returns:
        dist (torch.Tensor): distances [Vs, k]
        indices (torch.Tensor): indices [Vs, k]
    """
    assert method in ['brute', 'cpu_kd'], f'Invalid method: {method}, only supports "brute" or "cpu_kd"'
    if omit_diagonal and src_pts.shape[0] != target_pts.shape[0]:
        raise ValueError('omit_diagonal can only be used when source and target are the same shape')

    # use 'cpu_kd' for large points
    if src_pts.shape[0] * target_pts.shape[0] > 1e8:
        method = 'cpu_kd'

    if method == 'brute':
        # Expand so both are VsxVtx3 tensor
        src_pts_expand = src_pts.unsqueeze(1).expand(-1, target_pts.shape[0], -1)
        target_pts_expand = target_pts.unsqueeze(0).expand(src_pts.shape[0], -1, -1)

        # Compute distance between target points and source points
        dist_mat = norm(src_pts_expand - target_pts_expand)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        dist, indices = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return dist, indices
    else: # 'cpu_kd'
        assert largest == False, 'cannot do largest with cpu_kd'

        src_pts_np = torch2np(src_pts)
        target_pts_np = torch2np(target_pts)

        # Build the kd-tree
        kd_tree = neighbors.KDTree(target_pts_np)

        k_search = k + 1 if omit_diagonal else k
        _, indices = kd_tree.query(src_pts_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = indices != np.arange(indices.shape[0])[:, None]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            indices = indices[mask].reshape((indices.shape[0], indices.shape[1]-1))

        indices = torch.tensor(indices, device=src_pts.device, dtype=torch.int64)
        dist = norm(src_pts.unsqueeze(1).expand(-1, k, -1) - target_pts[indices])

        return dist, indices


def build_targent_frames(verts, faces, vert_normals=None):
    """
    Build targent frames for each vertices with three orthogonal basis.
    Args:
        verts (torch.Tensor): vertices [V, 3].
        faces (torch.Tensor): faces [F, 3]
        vert_normals (torch.Tensor, optional): vertex normals [V, 3]. Default None
    Return:
        frames (torch.Tensor): frames [V, 3, 3]
    """
    V = verts.shape[0]
    device = verts.device
    dtype = verts.dtype

    # compute vertex normals when necessary
    if not vert_normals:
        vert_normals = vertex_normal(verts, faces)

    # find an orthogonal basis
    basis_cand1 = torch.tensor([1, 0, 0], device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1, keepdim=True)) < 0.9), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def build_grad_point_cloud(verts, frames, n_neighbors=30):
    """
    Build gradient matrix for point cloud
    Args:
        verts (torch.Tensor): vertices [V, 3].
        frames (torch.Tensor): frames [V, 3, 3].
        n_neighbors (int, optional): number of neighbors. Default 30.
    Returns:

    """
    verts_np = torch2np(verts)

    # find neighboring points
    _, neigh_inds = find_knn(verts, verts, n_neighbors, omit_diagonal=True, method='cpu_kd')

    # build edges
    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors)
    edges = np.stack((edge_inds_from, torch2np(neigh_inds).flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad(verts_np, edges, torch2np(edge_tangent_vecs))


def edge_tangent_vectors(verts, frames, edges):
    """
    Compute edge tangent vectors
    Args:
        verts (torch.Tensor): vertices [V, 3].
        frames (torch.Tensor): frames [V, 3, 3].
        edges (torch.Tensor): edges [2, E], where E = V * k, k: number of nearest neighbor.
    Returns:
        egde_tangent (torch.Tensor): edge tangent vectors [E, 2].
    """
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator.
    Given real inputs at vertices,
    produces a complex (vector value) at vertices giving the gradient.

    Args:
        verts (np.ndarray): vertices [V, 3]
        edges (np.ndarray): edges [2, E]
        edge_tangent_vectors (np.ndarray): edge tangent vectors [E, 2]
    """

    # Build outgoining neighbor lists
    V = verts.shape[0]
    vert_edge_outgoing = [[] for _ in range(V)]
    for e in range(edges.shape[1]):
        tail_ind = edges[0, e]
        tip_ind = edges[1, e]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(e)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iv in range(V):
        n_neigh = len(vert_edge_outgoing[iv])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iv]
        for i_neigh in range(n_neigh):
            ie = vert_edge_outgoing[iv][i_neigh]
            jv = edges[1, ie]
            ind_lookup.append(jv)

            edge_vec = edge_tangent_vectors[ie][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iv)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)),
        shape=(V, V)
    ).tocsc()

    return mat


def laplacian_decomposition(verts, faces, k=150):
    """
    Laplacian decomposition
    Args:
        verts (np.ndarray): vertices [V, 3].
        faces (np.ndarray): faces [F, 3]
        k (int, optional): number of eigenvalues/vectors to compute. Default 120.

    Returns:
        - evals: (k) list of eigenvalues of the Laplacian matrix.
        - evecs: (V, k) list of eigenvectors of the Laplacian.
        - evecs_trans: (k, V) list of pseudo inverse of eigenvectors of the Laplacian.
    """
    assert k >= 0, f'Number of eigenvalues/vectors should be non-negative, bug get {k}'
    is_cloud = (faces is None)
    eps = 1e-8

    # Build Laplacian matrix
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts)
        massvec = M.diagonal()
    else:
        L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
        massvec = pp3d.vertex_areas(verts, faces)
        massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec).any():
        raise RuntimeError("NaN mass matrix")

    # Compute the eigenbasis
    # Prepare matrices
    L_eigsh = (L + eps * scipy.sparse.identity(L.shape[0])).tocsc()
    massvec_eigsh = massvec
    Mmat = scipy.sparse.diags(massvec_eigsh)
    eigs_sigma = eps

    fail_cnt = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)
            # Clip off any eigenvalues that end up slightly negative due to numerical error
            evals = np.clip(evals, a_min=0., a_max=float('inf'))
            evals = evals.reshape(-1, 1)
            break
        except:
            if fail_cnt > 3:
                raise ValueError('Failed to compute eigen-decomposition')
            fail_cnt += 1
            print('Decomposition failed; adding eps')
            L_eigsh = L_eigsh + (eps * 10 ** fail_cnt) * scipy.sparse.identity(L.shape[0])

    evecs = np.array(evecs, ndmin=2)
    evecs_trans = evecs.T @ Mmat

    sqrt_area = np.sqrt(Mmat.diagonal().sum())
    return evals, evecs, evecs_trans, sqrt_area


def compute_operators(verts, faces, k=120, normals=None):
    """
    Build spectral operators for a mesh/point cloud.
    Constructs mass matrix, eigenvalues/vectors for Laplacian,
    and gradient matrix.

    Args:
         verts (torch.Tensor): vertices [V, 3].
         faces (torch.Tensor): faces [F, 3]
         k (int, optional): number of eigenvalues/vectors to compute. Default 120.
         normals (torch.Tensor, optional): vertex normals [V, 3]. Default None

    Returns:
        spectral_operators (dict):
            - frames: (V, 3, 3) X/Y/Z coordinate frame at each vertex.
            - massvec: (V) real diagonal of lumped mass matrix.
            - L: (V, V) Laplacian matrix.
            - evals: (k) list of eigenvalues of the Laplacian matrix.
            - evecs: (V, k) list of eigenvectors of the Laplacian.
            - gradX: (V, V) sparse matrix which gives X-component of gradient in the local basis.
            - gradY: (V, V) same as gradX but for Y-component of gradient.

    Note: PyTorch doesn't seem to like complex sparse matrices,
    so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately,
    rather than as one complex sparse matrix.
    """
    assert k >= 0, f'Number of eigenvalues/vectors should be non-negative, bug get {k}'
    device = verts.device
    dtype = verts.dtype
    is_cloud = (faces is None)

    eps = 1e-8

    verts_np = torch2np(verts).astype(np.float64)
    faces_np = torch2np(faces) if faces is not None else None
    frames = build_targent_frames(verts, faces, vert_normals=normals)

    # Build Laplacian matrix
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()
    else:
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec_np).any():
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # Compute the eigenbasis
    if k > 0:
        # Prepare matrices
        L_eigsh = (L + eps * scipy.sparse.identity(L.shape[0])).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        fail_cnt = 0
        while True:
            try:
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)
                # Clip off any eigenvalues that end up slightly negative due to numerical error
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

                break
            except:
                if fail_cnt > 3:
                    raise ValueError('Failed to compute eigen-decomposition')
                fail_cnt += 1
                print('Decomposition failed; adding eps')
                L_eigsh = L_eigsh + (eps * 10 ** fail_cnt) * scipy.sparse.identity(L.shape[0])
    else: # k == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0], 0))

    # Build gradient matrices
    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts_np, torch2np(edges), torch2np(edge_vecs))

    # split complex gradient into two real sparse matrices (PyTorch doesn't like complex sparse matrix)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # convert to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return frames, massvec, L, evals, evecs, gradX, gradY


def get_operators(verts, faces, k=120, normals=None,
                  cache_dir=None, overwrite_cache=False):
    """
    See documentation for compute_operators().
    This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability,
    then truncated to single precision floats to store on disk,
    and finally returned as a tensor with dtype/device matching the `verts` input.
    """
    assert verts.dim() == 2, 'Please call get_all_operators() for a batch of vertices'
    device = verts.device
    dtype = verts.dtype
    verts_np = torch2np(verts)
    faces_np = torch2np(faces) if faces is not None else None

    if np.isnan(verts_np).any():
        raise ValueError('detect NaN vertices.')

    found = False
    if cache_dir:
        assert osp.isdir(cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))

        # Search through buckets with matching hashes.
        # When the loop exits,
        # this is the bucket index of the file we should write to.
        i_cache = 0
        while True:
            # From the name of the file to check
            search_path = osp.join(cache_dir, hash_key_str+'_'+str(i_cache)+'.npz')

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile['verts']
                cache_faces = npzfile['faces']
                cache_k = npzfile['k_eig'].item()

                # If the cache doesn't match, keep searching
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache += 1
                    print('collision detected')
                    continue

                # Delete previous file and overwrite it
                if overwrite_cache or cache_k < k:
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + '_data']
                    indices = npzfile[prefix + '_indices']
                    indptr = npzfile[prefix + '_indptr']
                    shape = npzfile[prefix + '_shape']
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # this entry matches. return it.
                frames = npzfile['frames']
                mass = npzfile['mass']
                L = read_sp_mat('L')
                evals = npzfile['evals'][:k]
                evecs = npzfile['evecs'][:, :k]
                gradX = read_sp_mat('gradX')
                gradY = read_sp_mat('gradY')

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = sparse_np_to_torch(gradY).to(device=device, dtype=dtype)

                found = True
                break
            except FileNotFoundError:
                # not found, create a new file
                break

    if not found:
        # recompute
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, k, normals)

        dtype_np = np.float32

        # save
        if cache_dir:
            frames_np = torch2np(frames).astype(dtype_np)
            mass_np = torch2np(mass).astype(dtype_np)
            evals_np = torch2np(evals).astype(dtype_np)
            evecs_np = torch2np(evecs).astype(dtype_np)
            L_np = sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)

            np.savez(
                search_path,
                verts=verts_np,
                faces=faces_np,
                k_eig=k,
                frames=frames_np,
                mass=mass_np,
                evals=evals_np,
                evecs=evecs_np,
                L_data=L_np.data,
                L_indices=L_np.indices,
                L_indptr=L_np.indptr,
                L_shape=L_np.shape,
                gradX_data=gradX_np.data,
                gradX_indices=gradX_np.indices,
                gradX_indptr=gradX_np.indptr,
                gradX_shape=gradX_np.shape,
                gradY_data=gradY_np.data,
                gradY_indices=gradY_np.indices,
                gradY_indptr=gradY_np.indptr,
                gradY_shape=gradY_np.shape,
            )

    return frames, mass, L, evals, evecs, gradX, gradY


def get_all_operators(verts, faces, k=120,
                      normals=None,
                    cache_dir=None):
    """
    Get all operators from batch
    """
    assert verts.dim() == 3, 'please call get_operators() for a single vertices'

    B = verts.shape[0]

    frames = []
    mass = []
    L = []
    evals = []
    evecs = []
    gradX = []
    gradY = []

    for i in range(B):
        if faces is not None:
            if normals is not None:
                output = get_operators(verts[i], faces[i], k, normals[i], cache_dir)
            else:
                output = get_operators(verts[i], faces[i], k, None, cache_dir)
        else:
            if normals is not None:
                output = get_operators(verts[i], None, k, normals[i], cache_dir)
            else:
                output = get_operators(verts[i], None, k, None, cache_dir)
        frames += [output[0]]
        mass += [output[1]]
        L += [output[2]]
        evals += [output[3]]
        evecs += [output[4]]
        gradX += [output[5]]
        gradY += [output[6]]

    frames = torch.stack(frames)
    mass = torch.stack(mass)
    L = torch.stack(L)
    evals = torch.stack(evals)
    evecs = torch.stack(evecs)
    gradX = torch.stack(gradX)
    gradY = torch.stack(gradY)

    return frames, mass, L, evals, evecs, gradX, gradY


def compute_hks_autoscale(evals, evecs, count=16):
    """
    Compute heat kernel signature with auto-scale
    Args:
        evals (torch.Tensor): eigenvalues of Laplacian matrix [B, K]
        evecs (torch.Tensor): eigenvecetors of Laplacian matrix [B, V, K]
        count (int, optional): number of hks. Default 16.
    Returns:
        out (torch.Tensor): heat kernel signature [B, V, count]
    """
    scales = torch.logspace(-2.0, 0.0, steps=count, device=evals.device, dtype=evals.dtype)

    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # [B, 1, S, K]
    terms = power_coefs * (evecs * evecs).unsqueeze(2) # [B, V, S, K]

    out = torch.sum(terms, dim=-1) # [B, V, S]

    return out


def wks(evals, evecs, energy_list, sigma, scaled=False):
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    indices = (evals > 1e-5)
    evals = evals[indices]
    evecs = evecs[:, indices]

    coefs = torch.exp(-torch.square(energy_list[:, None] - torch.log(torch.abs(evals))[None, :]) / (2 * sigma ** 2))

    weighted_evecs = evecs[None, :, :] * coefs[:, None, :]
    wks = torch.einsum('tnk,nk->nt', weighted_evecs, evecs)

    if scaled:
        inv_scaling = coefs.sum(1)
        return (1 / inv_scaling)[None, :] * wks
    else:
        return wks


def auto_wks(evals, evecs, n_descr, scaled=True):
    abs_ev = torch.sort(evals.abs())[0]
    e_min, e_max = torch.log(abs_ev[1]), torch.log(abs_ev[-1])
    sigma = 7 * (e_max - e_min) / n_descr

    e_min += 2 * sigma
    e_max -= 2 * sigma

    energy_list = torch.linspace(float(e_min), float(e_max), n_descr, device=evals.device, dtype=evals.dtype)

    return wks(abs_ev, evecs, energy_list, sigma, scaled=scaled)


def compute_wks_autoscale(evals, evecs, mass, n_descr=128, subsample_step=1, n_eig=128):
    feats = []
    for b in range(evals.shape[0]):
        feat = auto_wks(evals[b, :n_eig], evecs[b, :, :n_eig], n_descr, scaled=True)
        feat = feat[:, torch.arange(0, feat.shape[1], subsample_step)]
        feat_norm = torch.einsum('np,np->p', feat, mass[b].unsqueeze(1) * feat)
        feat /= torch.sqrt(feat_norm)
        feats += [feat]
    feats = torch.stack(feats, dim=0)
    return feats


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

    matrices = [R_x, R_y, R_z]

    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_random_rotation(x, y, z):
    thetas = torch.zeros(3, dtype=torch.float)
    degree_angles = [x, y, z]
    for axis_ind, deg_angle in enumerate(degree_angles):
        rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
        rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
        thetas[axis_ind] = rand_radian_angle

    return euler_angles_to_rotation_matrix(thetas)


def data_augmentation(verts, rot_x=0, rot_y=90.0, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # random rotation
    rotation_matrix = get_random_rotation(rot_x, rot_y, rot_z).repeat(verts.shape[0], 1, 1).to(verts.device)
    verts = torch.bmm(verts, rotation_matrix.transpose(1, 2))

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts


def procrustes(X, Y, scaling=False, reflection='best'):
    """
    Taken from https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform