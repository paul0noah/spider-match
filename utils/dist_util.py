# distributed training utils support for DistributedDataParallel (ddp)
import functools
import os
import re

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(backend='nccl', port=29500):
    """Initialize slurm distributed training environment.
    Args:
        backend (str, optional): Backend of torch.distributed. Default 'nccl'.
        port (int, optional): the port number for tcp/ip communication. Default 29500.
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    _init_dist_slurm(backend, port)


def _init_dist_slurm(backend, port):
    # 1. get environment info
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    node_list = str(os.environ['SLURM_NODELIST'])

    # 2. specify ip address
    node_parts = re.findall('[0-9]+', node_list)
    host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])

    # 3. specify port number
    port = str(port)

    # 4. initialize tcp/ip communication
    init_method = 'tcp://{}:{}'.format(host_ip, port)
    try:
        dist.init_process_group(backend, init_method=init_method, world_size=world_size, rank=rank)
    except:
        raise ValueError(f'Initialize DDP failed. The port {port} is already used. Please assign a different port.')

    # 5. specify current device
    torch.cuda.set_device(local_rank)


def master_only(func):
    """
    Function only executes in the master rank (rank = 0).

    Args:
        func (Callable): callable function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # get rank
        rank, _ = get_dist_info()

        # execute only when rank = 0
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def get_dist_info():
    """
    Get distribution information.

    Returns:
        rank (int): the rank number of current process group.
        world_size (int): the total number of the processes.
    """
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size
