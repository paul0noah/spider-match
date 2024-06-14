import os
import os.path as osp
import random
import time
import numpy as np
import torch

from .dist_util import master_only


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


@master_only
def mkdir_and_rename(path):
    """
    Make directory. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt['experiments_root'])
        os.makedirs(path_opt['models'], exist_ok=True)
        os.makedirs(path_opt['log'], exist_ok=True)
    else:
        mkdir_and_rename(path_opt['results_root'])
        os.makedirs(path_opt['visualization'], exist_ok=True)
        os.makedirs(path_opt['log'], exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file siz.
    """
    for unit in ['B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'


def log1p_safe(x):
  """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))
def expm1_safe(x):
  """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))

def robust_lossfun(x, alpha, scale, approximate=False, epsilon=1e-6):
  r"""Implements the general form of the loss.

  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.

  Args:
    x: The residual for which the loss is being computed. x can have any shape,
      and alpha and scale will be broadcasted to match x's shape if necessary.
      Must be a tensor of floats.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Must be a tensor of floats with the
      same precision as `x`. Varying alpha allows
      for smooth interpolation between a number of discrete robust losses:
      alpha=-Infinity: Welsch/Leclerc Loss.
      alpha=-2: Geman-McClure loss.
      alpha=0: Cauchy/Lortentzian loss.
      alpha=1: Charbonnier/pseudo-Huber loss.
      alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha. Must be a tensor of single-precision
      floats.
    approximate: a bool, where if True, this function returns an approximate and
      faster form of the loss, as described in the appendix of the paper. This
      approximation holds well everywhere except as x and alpha approach zero.
    epsilon: A float that determines how inaccurate the "approximate" version of
      the loss will be. Larger values are less accurate but more numerically
      stable. Must be great than single-precision machine epsilon.

  Returns:
    The losses for each element of x, in the same shape and precision as x.
  """
  assert torch.is_tensor(x)
  assert torch.is_tensor(scale)
  assert torch.is_tensor(alpha)
  assert alpha.dtype == x.dtype
  assert scale.dtype == x.dtype
  assert (scale > 0).all()
  if approximate:
    # `epsilon` must be greater than single-precision machine epsilon.
    assert epsilon > np.finfo(np.float32).eps
    # Compute an approximate form of the loss which is faster, but innacurate
    # when x and alpha are near zero.
    b = torch.abs(alpha - 2) + epsilon
    d = torch.where(alpha >= 0, alpha + epsilon, alpha - epsilon)
    loss = (b / d) * (torch.pow((x / scale)**2 / b + 1., 0.5 * d) - 1.)
  else:
    # Compute the exact loss.

    # This will be used repeatedly.
    squared_scaled_x = (x / scale)**4

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(x)
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.max(machine_epsilon, torch.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(alpha >= 0, torch.ones_like(alpha),
                             -torch.ones_like(alpha)) * torch.max(
                                 machine_epsilon, torch.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == -float('inf'), loss_neginf,
        torch.where(
            alpha == 0, loss_zero,
            torch.where(
                alpha == 2, loss_two,
                torch.where(alpha == float('inf'), loss_posinf,
                            loss_otherwise))))

  return loss