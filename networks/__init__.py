import importlib
import os.path as osp

from utils import get_root_logger, scandir
from utils.registry import NETWORK_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_network.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'networks.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    """Build network from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Network type.

    Returns:
        network (nn.Module): network built by opt.
    """
    network_type = opt.pop('type')
    network = NETWORK_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{network.__class__.__name__}] is created.')
    return network
