import numpy as np
import torch


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p


def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)


def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21

