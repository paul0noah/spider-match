import scipy.io
from utils.fps_approx import *
from utils.geometry_util import *
import numpy as np
import pymeshfix
import igl
from sklearn.neighbors import NearestNeighbors
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from networks.diffusion_network import DiffusionNet
from networks.permutation_network import Similarity
from networks.fmap_network import RegularizedFMNet

from utils.shape_util import read_shape
from utils.geometry_util import compute_operators
from utils.fmap_util import nn_query, fmap2pointmap
from utils.tensor_util import to_numpy

from losses.fmap_loss import SURFMNetLoss, PartialFmapsLoss, SquaredFrobeniusLoss
from losses.dirichlet_loss import DirichletLoss


def compute_features(vert_x, face_x, vert_y, face_y, feature_extractor, normalize=True):
    feat_x = feature_extractor(vert_x.unsqueeze(0), face_x.unsqueeze(0))
    feat_y = feature_extractor(vert_y.unsqueeze(0), face_y.unsqueeze(0))
    # normalize features
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)

    return feat_x, feat_y

def compute_permutation_matrix(feat_x, feat_y, permutation, bidirectional=False, normalize=True):
    # normalize features
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
    similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

    Pxy = permutation(similarity)

    if bidirectional:
        Pyx = permutation(similarity.transpose(1, 2))
        return Pxy, Pyx
    else:
        return Pxy

def update_network(loss_metrics, feature_extractor, optimizer):
    # compute total loss
    loss = 0.0
    for k, v in loss_metrics.items():
        if k != 'l_total':
            loss += v
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    # clip gradient for stability
    torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
    # update weight
    optimizer.step()

    return loss

def get_feature_extractor(network_path, input_type='wks', num_refine=0, partial=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_channels = 128 if input_type == 'wks' else 3  # 'xyz'
    feature_extractor = DiffusionNet(in_channels=in_channels, out_channels=256, input_type=input_type).to(device)
    feature_extractor.load_state_dict(torch.load(network_path, map_location=torch.device(device))['networks']['feature_extractor'], strict=True)
    return feature_extractor

def get_opts_for_faust():
    network_path = "checkpoints/faust.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 0,
        "partial": False,
        "non_isometric": False,
    }
    return feature_opts

def get_opts_for_dt4d_inter():
    network_path = "checkpoints/dt4d.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 15,
        "partial": False,
        "non_isometric": True,
    }
    return feature_opts

def get_opts_for_dt4d_intra():
    network_path = "checkpoints/dt4d.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 0,
        "partial": False,
        "non_isometric": False,
    }
    return feature_opts

def get_opts_for_smal():
    network_path = "checkpoints/smal.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 15,
        "partial": False,
        "non_isometric": True,
    }
    return feature_opts

def get_feature_opts(dataset):
    if 'faust' in dataset.lower():
        return get_opts_for_faust()
    if 'smal' in dataset.lower():
        return get_opts_for_smal()
    if 'dt4d_inter' in dataset.lower():
        return get_opts_for_dt4d_inter()
    if 'dt4d_intra' in dataset.lower():
        return get_opts_for_dt4d_intra()

def shape_loader(filename1, filename2, shape_loader_opts):
    vert_np_x, face_np_x = read_shape(filename1)
    vert_np_y, face_np_y = read_shape(filename2)
    vert_np_x -= np.mean(vert_np_x, axis=0)
    vert_np_y -= np.mean(vert_np_y, axis=0)

    VX_orig = vert_np_x
    FX_orig = face_np_x
    VY_orig = vert_np_y
    FY_orig = face_np_y
    VX, FX = pymeshfix.clean_from_arrays(VX_orig, FX_orig)
    VY, FY = pymeshfix.clean_from_arrays(VY_orig, FY_orig)
    nfacesX = min(shape_loader_opts['num_faces'], len(FX_orig))
    nfacesY = min(shape_loader_opts['num_faces'], len(FY_orig))
    partial = False
    if "partial" in shape_loader_opts:
        partial = shape_loader_opts['partial']
    if partial:
        ratio = shape_loader_opts['num_faces'] / FX.shape[0]
        nfacesY = round(FY.shape[0] * ratio)

    _, VX, FX, _, _ = igl.decimate(VX, FX, nfacesX)
    _, VY, FY, _, _ = igl.decimate(VY, FY, nfacesY)

    idx_vx_in_orig = knn_search(VX, VX_orig)
    if partial:
        idx_in_partial = knn_search(VY_orig, VY)
        idx_vy_in_orig = np.zeros((idx_in_partial.size, 2), dtype=int)
        idx_vy_in_orig[:, 0] = idx_in_partial
        idx_vy_in_orig[:, 1] = knn_search(VY[idx_in_partial, :], VY_orig)
    else:
        idx_vy_in_orig = knn_search(VY, VY_orig)

    return VX_orig, FX_orig, VX, FX, idx_vx_in_orig, VY_orig, FY_orig, VY, FY, idx_vy_in_orig

def get_features(VX, FX, VY, FY, feature_opts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # convert numpy to tensor
    vert_x, face_x = to_tensor(VX, FX, device)
    vert_y, face_y = to_tensor(VY, FY, device)

    # compute Laplacian
    _, mass_x, Lx, evals_x, evecs_x, _, _ = compute_operators(vert_x, face_x, k=200)
    _, mass_y, Ly, evals_y, evecs_y, _, _ = compute_operators(vert_y, face_y, k=200)
    evecs_trans_x = evecs_x.T * mass_x[None]
    evecs_trans_y = evecs_y.T * mass_y[None]
    feature_extractor = feature_opts['feature_extractor']
    feature_extractor.eval()
    num_refine = feature_opts['num_refine']
    partial = feature_opts['partial']
    non_isometric = feature_opts['non_isometric']
    if num_refine > 0:
        with torch.set_grad_enabled(True):
            permutation = Similarity(tau=0.07, hard=False).to(device)
            fmap_net = RegularizedFMNet(bidirectional=True)
            optimizer = optim.Adam(feature_extractor.parameters(), lr=1e-3)
            fmap_loss = SURFMNetLoss(w_bij=1.0, w_orth=1.0, w_lap=0.0) if not partial else PartialFmapsLoss(w_bij=1.0, w_orth=1.0)
            align_loss = SquaredFrobeniusLoss(loss_weight=1.0)
            if non_isometric:
                w_dirichlet = 5.0
            else:
                if partial:
                    w_dirichlet = 1.0
                else:
                    w_dirichlet = 0.0
            dirichlet_loss = DirichletLoss(loss_weight=w_dirichlet)
            print('Test-time adaptation')
            pbar = tqdm(range(num_refine))
            for _ in pbar:
                feat_x, feat_y = compute_features(vert_x, face_x, vert_y, face_y, feature_extractor, normalize=False)
                Cxy, Cyx = fmap_net(feat_x, feat_y, evals_x.unsqueeze(0), evals_y.unsqueeze(0),
                                    evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0))
                Pxy, Pyx = compute_permutation_matrix(feat_x, feat_y, permutation, bidirectional=True, normalize=True)

                # compute functional map regularisation loss
                loss_metrics = fmap_loss(Cxy, Cyx, evals_x.unsqueeze(0), evals_y.unsqueeze(0))
                # compute C
                Cxy_est = torch.bmm(evecs_trans_y.unsqueeze(0), torch.bmm(Pyx, evecs_x.unsqueeze(0)))

                # compute couple loss
                loss_metrics['l_align'] = align_loss(Cxy, Cxy_est)
                if not partial:
                    Cyx_est = torch.bmm(evecs_trans_x.unsqueeze(0), torch.bmm(Pxy, evecs_y.unsqueeze(0)))
                    loss_metrics['l_align'] += align_loss(Cyx, Cyx_est)

                # compute dirichlet energy
                if non_isometric:
                    loss_metrics['l_d'] = (dirichlet_loss(torch.bmm(Pxy, vert_y.unsqueeze(0)), Lx.to_dense().unsqueeze(0)) +
                                           dirichlet_loss(torch.bmm(Pyx, vert_x.unsqueeze(0)), Ly.to_dense().unsqueeze(0)))

                loss = update_network(loss_metrics, feature_extractor, optimizer)
                pbar.set_description(f'Total loss: {loss:.4f}')

    feature_extractor.eval()
    with torch.no_grad():
        feat_x, feat_y = compute_features(vert_x, face_x, vert_y, face_y, feature_extractor)

    feat_x = to_numpy(feat_x)
    feat_y = to_numpy(feat_y)

    return feat_x, feat_y

@torch.no_grad()
def get_siggraph_matching(VX_orig, FX_orig, VX, FX, idx_vx_in_orig, VY_orig, FY_orig, VY, FY, idx_vy_in_orig, feature_opts, non_isometric=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat_x, feat_y = get_features(VX_orig, FX_orig, VY_orig, FY_orig, feature_opts)
    feat_x = feat_x[idx_vx_in_orig, :]
    if feature_opts["partial"]:
        #remove closing triangles
        isin0 = np.isin(FY[:, 0], idx_vy_in_orig[:, 0]).astype(int)
        isin1 = np.isin(FY[:, 1], idx_vy_in_orig[:, 0]).astype(int)
        isin2 = np.isin(FY[:, 2], idx_vy_in_orig[:, 0]).astype(int)
        FY = FY[(isin0 + isin1 + isin2) >= 3, :]
        NVY, FY, _, _ = igl.remove_unreferenced(VY, FY)
        nvy2vy = knn_search(NVY, VY)
        VY = NVY
        feat_y = feat_y[nvy2vy, :]
    else:
        feat_y = feat_y[idx_vy_in_orig, :]
    feat_x = torch.from_numpy(feat_x).to(device=device, dtype=torch.float32)
    feat_y = torch.from_numpy(feat_y).to(device=device, dtype=torch.float32)
    # convert numpy to tensor
    vert_x, face_x = to_tensor(VX, FX, device)
    vert_y, face_y = to_tensor(VY, FY, device)

    # compute Laplacian
    _, mass_x, Lx, evals_x, evecs_x, _, _ = compute_operators(vert_x, face_x, k=200)
    _, mass_y, Ly, evals_y, evecs_y, _, _ = compute_operators(vert_y, face_y, k=200)
    evecs_trans_x = evecs_x.T * mass_x[None]
    evecs_trans_y = evecs_y.T * mass_y[None]
    p2p = get_p2p(feat_x.reshape((1, feat_x.shape[0], feat_x.shape[1])), evecs_x, evecs_trans_x,
                  feat_y.reshape((1, feat_y.shape[0], feat_y.shape[1])), evecs_y, evecs_trans_y, non_isometric, device)

    if feature_opts["partial"]:
        p2p[:, 1] = nvy2vy[p2p[:, 1]]
    else:
        return p2p

def get_fastdog_opts(solvertype):
    fastdogopts = {
        "omega": 0.5,
        "num_dual_iters": 2000,
        "rel_improvement_slope": 1e-6,
        "cuda_split_long_bdds": False,
        "non_learned_lbfgs_step_size": 1e-5,
        "init_delta": 1.1,
        "delta_growth_rate": 1.1,
        "num_dual_iters_primal_rounding": 500,
        "num_primal_rounds": 200,
    }
    if solvertype == "fastdog":
        fastdogopts["cuda_split_long_bdds"] = False
    return fastdogopts

def get_augmented_edge(vy, fy):
    VF, NI = igl.vertex_triangle_adjacency(fy, fy.max()+1)
    adj_fy_fy, _ = igl.triangle_triangle_adjacency(fy)
    angles_fy = igl.internal_angles(vy, fy)
    is_non_obtuse = np.all(angles_fy < np.pi, axis=1)
    num_new_edges = 0
    new_edges = np.zeros((6 * fy.shape[0], 2), dtype=int)
    for vert in range(0, vy.shape[0]):
        tri_idxs = VF[range(NI[vert], NI[vert+1])]
        for tri_idx in tri_idxs:
            if not is_non_obtuse[tri_idx]:
                continue
            tri_nei_idxs = adj_fy_fy[tri_idx]
            is_vertex_adjacent = [True] * 3
            for i in range(3):
                is_vertex_adjacent[i] = tri_nei_idxs[i] in tri_idxs
            tri_nei_idx = tri_nei_idxs[np.logical_not(is_vertex_adjacent)][0]
            if not is_non_obtuse[tri_nei_idx]:
                continue

            new_edge = np.setxor1d(fy[tri_idx], fy[tri_nei_idx])
            if np.any(np.linalg.norm(new_edges[:num_new_edges] - new_edge, axis=1) == 0):
                # already exists
                continue
            assert(new_edge.shape[0] == 2)
            new_edges[num_new_edges] = new_edge
            num_new_edges = num_new_edges + 1
            new_edges[num_new_edges] = new_edge[[1, 0]]
            num_new_edges = num_new_edges + 1
    new_edges = new_edges[:num_new_edges]
    ey = igl.edges(fy)
    ey = np.append(ey, ey[:, [1, 0]], axis=0)
    return np.unique(np.row_stack((new_edges, ey)), axis=1)

def get_spider_curve(vx, fx, vy, fy):
    ey = get_augmented_edge(vy, fy)

    print("Solving salesman problem...")
    total_path_indices = metric_salesman_approx(vx, fx, augmented_edges=True)
    print("Done")

    ex = np.zeros((len(total_path_indices), 2), dtype='int')
    ex[:, 0] = total_path_indices
    ex[:-1, 1] = ex[1:, 0]
    ex[-1, 1] = ex[0, 0]

    return ey, ex

def load_contour(general_opts):
    status = {}
    status["success"] = False
    if not "export_file" in general_opts:
        return status
    if os.path.isfile(general_opts["export_file"]):
        mdic = scipy.io.loadmat(general_opts["export_file"])
        status["ex"] = mdic["X"]["edges"][0][0] - 1
        status["sx"] = mdic["Sx"][0] - 1
        status["success"] = True
    return status


def sort_matching(matching, ex, skip_presort=False):
    # presort
    if not skip_presort:
        num_back = 1
        while matching[-1, 0] == matching[-1 - num_back, 0]:
            num_back += 1
        matching = np.row_stack((matching[-num_back:, :], matching[:-num_back, :]))
    # sort the matching
    sorted_matching = np.zeros_like(matching)
    last_idx = 0
    num_added_to_sorted = 0
    for i, vertex_group in enumerate(ex[:, 0]):
        if i + 1 > ex.shape[0]:
            break
        next_vertex_group = ex[(i + 1) % ex.shape[0], 0]
        num_elements_in_group = 0
        while matching[last_idx + num_elements_in_group, 0] == vertex_group:
            num_elements_in_group += 1
            if last_idx + num_elements_in_group >= matching.shape[0]:
                break
        to_be_sorted = matching[last_idx:last_idx + num_elements_in_group, :]
        last_idx += num_elements_in_group
        # remove back forth edges
        is_back_forth = np.zeros(shape=(1, num_elements_in_group), dtype=bool).flatten()
        for ii in range(num_elements_in_group):
            if to_be_sorted[ii, 0] != to_be_sorted[ii, 1]:
                continue
            if is_back_forth[ii]:
                continue
            for jj in range(ii + 1, num_elements_in_group):
                if to_be_sorted[jj, 0] != to_be_sorted[jj, 1]:
                    continue
                if to_be_sorted[ii, 3] == to_be_sorted[jj, 2] and to_be_sorted[ii, 2] == to_be_sorted[jj, 3]:
                    is_back_forth[ii] = True
                    is_back_forth[jj] = True
                    break
        if np.any(is_back_forth):
            print("Removing back forth edges: " + str(is_back_forth.astype(int).sum()))
        to_be_sorted = to_be_sorted[np.logical_not(is_back_forth), :]
        # to_be_sorted = np.unique(to_be_sorted, axis=0)
        num_unique_elements = to_be_sorted.shape[0]
        sort_idx = -np.ones(shape=(1, num_unique_elements), dtype=int)
        used_elements = np.zeros(shape=(1, num_unique_elements), dtype=bool).flatten()

        # last element is easy
        try:
            sort_idx[0, -1] = np.argwhere(to_be_sorted[:, 1] != vertex_group)
        except:
            print("sort_idx to be empty")
        src3d = to_be_sorted[sort_idx[0, -1], 2]
        # rest is pain
        handled_degenerate3d = False
        for j in range(num_unique_elements - 2, -1, -1):
            next_idx = np.argwhere(np.logical_and(
                                        np.logical_and(to_be_sorted[:, 2] == src3d,
                                                       to_be_sorted[:, 3] == src3d),
                                        to_be_sorted[:, 0] == to_be_sorted[:, 1]))
            if not handled_degenerate3d and next_idx.shape[0] > 0:
                try:
                    sort_idx[0, j] = next_idx
                except:
                    print("error")
                used_elements[next_idx] = True
                handled_degenerate3d = True
                continue
            next_idx = np.argwhere(np.logical_and(to_be_sorted[:, 2] != src3d, to_be_sorted[:, 3] == src3d))
            if next_idx.shape[0] > 1:
                for nidx in next_idx:
                    if not used_elements[nidx]:
                        next_idx = nidx
                        break
            handled_degenerate3d = False
            try:
                sort_idx[0, j] = next_idx

                used_elements[next_idx] = True
                src3d = to_be_sorted[next_idx, 2].flatten()[0]
            except:
                print("Unhandled Error in sort edges")
        sorted_matching[num_added_to_sorted:num_added_to_sorted + num_unique_elements] = to_be_sorted[sort_idx, :]
        num_added_to_sorted += num_unique_elements
    return sorted_matching[:num_added_to_sorted]

def calc_geo_err(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map [points from X, points from Y]. shape [num matches]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    # Note: corr_x == u2x,  corr_y == u2y (u for universe), and p2p is x2y
    corr_y_inv = -np.ones(max(corr_y.max()+1, corr_y.shape[0]), dtype=np.int32)
    corr_y_inv[corr_y] = np.arange(corr_y.shape[0])

    y2u = corr_y_inv[p2p[:, 1]]
    ind_not_empty = y2u != -1

    y2x = corr_x[y2u[ind_not_empty]]

    ind21 = np.stack([p2p[ind_not_empty, 0], y2x], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err

def to_tensor(vert_np, face_np, device):
    vert = torch.from_numpy(vert_np).to(device=device, dtype=torch.float32)
    face = torch.from_numpy(face_np).to(device=device, dtype=torch.long)

    return vert, face

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

def get_p2p(feat_x, evecs_x, evecs_trans_x, feat_y, evecs_y, evecs_trans_y, non_isometric, device):
    if non_isometric:
        # nearest neighbour query
        p2p = nn_query(feat_x, feat_y).squeeze()
        # compute Pyx from functional map
        #Cxy = evecs_trans_y @ evecs_x[p2p]
        #Pyx = evecs_y @ Cxy @ evecs_trans_x
    else:
        # compute Pyx
        similarity = torch.bmm(feat_y, feat_x.transpose(1, 2))
        permutation = Similarity(tau=0.07, hard=True).to(device)
        Pyx = permutation(similarity).squeeze(0)
        Cxy = evecs_trans_y @ (Pyx @ evecs_x)
        # convert functional map to point-to-point map
        p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)
        # compute Pyx from functional map
        #Pyx = evecs_y @ Cxy @ evecs_trans_x
    p2p = to_numpy(p2p)
    p2p = np.stack([p2p, np.arange(feat_y.shape[1])], axis=-1)
    return p2p

def get_color_transfer(VX, VY, point_map):
    cmapX = VX
    cmapX = cmapX - cmapX.min(axis=0)
    cmapX = cmapX / cmapX.max(axis=0)
    cmapY = 0 * VY
    # cmapY[point_map[:, 1]] = cmap[point_map[:, 0]]
    idx = np.argsort(point_map[:, 1])
    sorted_pm = point_map[idx, 1]
    current_id = -1
    i = 0
    for pmi in sorted_pm:
        if pmi != current_id:
            if current_id != -1:
                cmapY[current_id] = cmapY[current_id] / num_elem
            num_elem = 1
            current_id = pmi
        else:
            num_elem = num_elem + 1
        cmapY[current_id] = cmapY[current_id] + cmapX[point_map[idx[i], 0]]
        i = i + 1
    return cmapX, cmapY
