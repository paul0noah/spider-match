import importlib
import os.path as osp

from utils import get_root_logger, scandir
from utils.registry import LOSS_REGISTRY

__all__ = ['build_loss']

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with
# '_loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# import all the model modules
_loss_modules = [importlib.import_module(f'losses.{file_name}') for file_name in loss_filenames]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Loss type.

    Returns:
        loss (nn.Module): loss built by opt.
    """
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss