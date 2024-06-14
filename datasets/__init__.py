import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

import torchvision.transforms as transforms

from utils import get_root_logger, scandir
from utils.dist_util import get_dist_info
from utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder, recursive=True) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'datasets.{file_name}') for file_name in dataset_filenames]


def build_transform(transform_opt):
    transform = []
    normalize = None # normalize can only be applied in torch.Tensor

    for name, params in transform_opt.items():
        if name == 'CenterCrop':
            transform += [transforms.CenterCrop(**params)]
        elif name == 'RandomCrop':
            transform += [transforms.RandomCrop(**params)]
        elif name == 'Resize':
            transform += [transforms.Resize(**params)]
        elif name == 'RandomResizedCrop':
            transform += [transforms.RandomResizedCrop(**params)]
        elif name == 'RandomHorizontalFlip':
            transform += [transforms.RandomHorizontalFlip(**params)]
        elif name == 'RandomVerticalFlip':
            transform += [transforms.RandomVerticalFlip(**params)]
        elif name == 'FiveCrop':
            transform += [transforms.FiveCrop(**params)]
        elif name == 'Pad':
            transform += [transforms.Pad(**params)]
        elif name == 'RandomAffine':
            transform += [transforms.RandomAffine(**params)]
        elif name == 'ColorJitter':
            transform += [transforms.ColorJitter(**params)]
        elif name == 'Grayscale':
            transform += [transforms.Grayscale(**params)]
        elif name == 'Normalize':
            normalize = params
        else:
            raise ValueError(f'The transform {name} is currently not support!')

    # convert PIL.Image to torch.Tensor
    transform += [transforms.ToTensor()]
    if normalize:
        transform += [transforms.Normalize(**normalize)]

    return transforms.Compose(transform)


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    Return:
        dataset (torch.utils.data.Dataset): dataset built by opt.
    """
    dataset_opt = deepcopy(dataset_opt)
    type = dataset_opt.pop('type')
    dataset_name = dataset_opt.pop('name')
    # build transform
    if 'transform' in dataset_opt:
        dataset_opt['transform'] = build_transform(dataset_opt.pop('transform'))
    # build dataset
    dataset = DATASET_REGISTRY.get(type)(**dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset [{dataset.__class__.__name__}]-[{dataset_name}] is built.')
    return dataset


def build_dataloader(dataset, dataset_opt, phase, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker (int): Number of workers for each GPU.
            batch_size (int): Training batch size for each GPU.
        phase (str): Phase. 'train' or 'val' or 'test'
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None

    Returns:
        dataloader (torch.utils.data.DataLoader): dataloader built by opt.
    """
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            denominator = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size']
            assert batch_size % denominator == 0, f'Batch size: {batch_size} must be divisible by {denominator}.'
            batch_size //= denominator
            num_workers = dataset_opt['num_worker']
        else:  # non-distributed training
            batch_size = dataset_opt['batch_size']
            num_workers = dataset_opt['num_worker']
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=2)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
