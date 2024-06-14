import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel

from networks import build_network
from losses import build_loss
from metrics import build_metric

from .lr_scheduler import MultiStepRestartLR
from utils import get_root_logger
from utils.dist_util import master_only
from utils.logger import AvgTimer
from utils.texture_util import write_obj_pair, write_point_cloud_pair
from utils.tensor_util import to_numpy


class BaseModel:
    """
    Base Model to be inherited by other models.
    Usually, feed_data, optimize_parameters and update_model
    need to be override.
    """

    def __init__(self, opt):
        """
        Construct BaseModel.
        Args:
             opt (dict): option dictionary contains all information related to model.
        """
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']

        # build networks
        self.networks = OrderedDict()
        self._setup_networks()

        # move networks to device
        self._networks_to_device()

        # print networks info
        self.print_networks()

        # setup metrics
        self.metrics = OrderedDict()
        self._setup_metrics()

        # training setting
        if self.is_train:
            # train mode
            self.train()
            # init optimizers, schedulers and losses
            self._init_training_setting()

        # load pretrained models
        load_path = self.opt['path'].get('resume_state')
        if load_path and os.path.isfile(load_path):
            state_dict = torch.load(load_path)
            if self.opt['path'].get('resume', True):  # resume training
                self.resume_model(state_dict, net_only=False)
            else:  # only resume model for validation
                self.resume_model(state_dict, net_only=True)

    def feed_data(self, data):
        """process data"""
        pass

    def optimize_parameters(self):
        """forward pass"""
        # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k != 'l_total':
                loss += v

        # update loss metrics
        self.loss_metrics['l_total'] = loss

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        loss.backward()

        # clip gradient for stability
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()

    def update_model_per_iteration(self):
        """update model per iteration"""
        for name in self.schedulers:
            if isinstance(self.schedulers[name], optim.lr_scheduler.OneCycleLR):
                self.schedulers[name].step()

    def update_model_per_epoch(self):
        """
        Update model per epoch.
        """
        for name in self.schedulers:
            if isinstance(self.schedulers[name], (optim.lr_scheduler.StepLR, optim.lr_scheduler.MultiStepLR,
                                                  MultiStepRestartLR, optim.lr_scheduler.ExponentialLR,
                                                  optim.lr_scheduler.CosineAnnealingLR,
                                                  optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                self.schedulers[name].step()

    def get_current_learning_rate(self):
        """
        Get current learning rate for each optimizer

        Returns:
            [list]: lis of learning rate
        """
        return [optimizer.param_groups[0]['lr'] for optimizer in self.optimizers.values()]

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            tb_logger (tensorboard logger): Tensorboard logger.
            update (bool): update best metric and best model. Default True
        """
        self.eval()

        # save results
        metrics_result = {}

        # geodesic errors
        geo_errors = []

        # one iteration
        timer = AvgTimer()
        pbar = tqdm(dataloader)
        for index, data in enumerate(pbar):
            p2p, Pyx, Cxy = self.validate_single(data, timer)
            p2p, Pyx, Cxy = to_numpy(p2p), to_numpy(Pyx), to_numpy(Cxy)
            if 'geo_error' in self.metrics:
                data_x, data_y = data['first'], data['second']
                # get geodesic distance matrix
                if 'dist' in data_x:
                    dist_x = data_x['dist']
                else:
                    dist_x = torch.cdist(data_x['verts'], data_x['verts'])

                # get gt correspondence
                corr_x = data_x['corr']
                corr_y = data_y['corr']

                # convert torch.Tensor to numpy.ndarray
                dist_x = to_numpy(dist_x)
                corr_x = to_numpy(corr_x)
                corr_y = to_numpy(corr_y)

                # compute geodesic error
                geo_err = self.metrics['geo_error'](dist_x, corr_x, corr_y, p2p, return_mean=False)

                # show result
                pbar.set_description(f'geo error: {geo_err.mean():.4f}')

                geo_errors += [geo_err]

            # visualize results
            if self.opt.get('visualize', False):
                data_x, data_y = data['first'], data['second']
                verts_x, verts_y = to_numpy(data_x['verts']), to_numpy(data_y['verts'])
                name_x, name_y = data['first']['name'][0], data['second']['name'][0]
                if 'faces' in data_x:
                    if os.path.isfile('figures/texture.png'):
                        shutil.copy('figures/texture.png',
                                    os.path.join(self.opt['path']['visualization'], 'texture.png'))
                    faces_x, faces_y = to_numpy(data_x['faces']), to_numpy(data_y['faces'])
                    file_x = os.path.join(self.opt['path']['visualization'], f'{name_x}.obj')
                    file_y = os.path.join(self.opt['path']['visualization'], f'{name_x}-{name_y}.obj')
                    write_obj_pair(file_x, file_y, verts_x, faces_x, verts_y, faces_y, Pyx, 'texture.png')
                else:
                    file_x = os.path.join(self.opt['path']['visualization'], f'{name_x}.ply')
                    file_y = os.path.join(self.opt['path']['visualization'], f'{name_y}-{name_x}.ply')
                    write_point_cloud_pair(file_x, file_y, verts_x, verts_y, p2p)

                # save functional map and point-wise correspondences
                save_dict = {'Cxy': Cxy, 'p2p': p2p + 1}  # plus one for MATLAB
                sio.savemat(os.path.join(self.opt['path']['visualization'], f'{name_x}-{name_y}.mat'), save_dict)

        logger = get_root_logger()
        logger.info(f'Avg time: {timer.get_avg_time():.4f}')

        if len(geo_errors) != 0:
            geo_errors = np.concatenate(geo_errors)
            avg_geo_error = geo_errors.mean()
            metrics_result['avg_error'] = avg_geo_error

            auc, fig, pcks = self.metrics['plot_pck'](geo_errors)
            metrics_result['auc'] = auc

            if tb_logger is not None:
                step = self.curr_iter // self.opt['val']['val_freq']
                tb_logger.add_figure('pck', fig, global_step=step)
                tb_logger.add_scalar('val auc', auc, global_step=step)
                tb_logger.add_scalar('val avg error', avg_geo_error, global_step=step)
            else:
                # save image
                plt.savefig(os.path.join(self.opt['path']['visualization'], 'pck.png'))
                # save pcks
                np.save(os.path.join(self.opt['path']['visualization'], 'pck.npy'), pcks)

            # display results
            logger = get_root_logger()
            logger.info(f'Val auc: {auc:.4f}')
            logger.info(f'Val avg error: {avg_geo_error:.4f}')

            # update best model state dict
            if update and (self.best_metric is None or (metrics_result['avg_error'] < self.best_metric)):
                self.best_metric = metrics_result['avg_error']
                self.best_networks_state_dict = self._get_networks_state_dict()
                logger.info(f'Best model is updated, average geodesic error: {self.best_metric:.4f}')

        # train mode
        self.train()

    def validate_single(self, data, timer):
        raise NotImplementedError

    def get_loss_metrics(self):
        if self.opt['dist']:
            self.loss_metrics = self._reduce_loss_dict()

        return self.loss_metrics

    def _init_training_setting(self):
        """
        Set up losses, optimizers and schedulers
        """
        # current epoch and iteration step
        self.curr_epoch = 0
        self.curr_iter = 0
        # optimizers and lr_schedulers
        self.optimizers = OrderedDict()
        self.schedulers = OrderedDict()

        # setup optimizers and schedulers
        self._setup_optimizers()
        self._setup_schedulers()

        # setup losses
        self.losses = OrderedDict()
        self._setup_losses()

        # loss metrics
        self.loss_metrics = OrderedDict()

        # best networks
        self.best_networks_state_dict = self._get_networks_state_dict()
        self.best_metric = None  # best metric to measure network

    def _setup_optimizers(self):
        def get_optimizer():
            if optim_type == 'Adam':
                return optim.Adam(optim_params, **train_opt['optims'][name])
            elif optim_type == 'AdamW':
                return optim.AdamW(optim_params, **train_opt['optims'][name])
            elif optim_type == 'RMSprop':
                return optim.RMSprop(optim_params, **train_opt['optims'][name])
            elif optim_type == 'SGD':
                return optim.SGD(optim_params, **train_opt['optims'][name])
            else:
                raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')

        train_opt = deepcopy(self.opt['train'])
        for name in self.networks:
            optim_params = []
            net = self.networks[name]
            for k, v in net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Parms {k} will not be optimized.')

            # no params to optimize
            if len(optim_params) == 0:
                logger = get_root_logger()
                logger.info(f'Network {name} has no param to optimize. Ignore it.')
                continue

            if name in train_opt['optims']:
                optim_type = train_opt['optims'][name].pop('type')
                optimizer = get_optimizer()
                self.optimizers[name] = optimizer
            else:
                # not optimize the network
                logger = get_root_logger()
                logger.warning(f'Network {name} will not be optimized.')

    def _setup_schedulers(self):
        """
        Set up lr_schedulers
        """
        train_opt = deepcopy(self.opt['train'])
        scheduler_opts = train_opt['schedulers']
        for name, optimizer in self.optimizers.items():
            scheduler_type = scheduler_opts[name].pop('type')
            if scheduler_type in 'MultiStepRestartLR':
                self.schedulers[name] = MultiStepRestartLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'CosineAnnealingLR':
                self.schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'CosineAnnealingWarmRestarts':
                self.schedulers[name] = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                       **scheduler_opts[name])
            elif scheduler_type == 'OneCycleLR':
                self.schedulers[name] = optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'StepLR':
                self.schedulers[name] = optim.lr_scheduler.StepLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'MultiStepLR':
                self.schedulers[name] = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'ExponentialLR':
                self.schedulers[name] = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'none':
                self.schedulers[name] = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def _setup_losses(self):
        """
        Setup losses
        """
        train_opt = deepcopy(self.opt['train'])
        if 'losses' in train_opt:
            for name, loss_opt in train_opt['losses'].items():
                self.losses[name] = build_loss(loss_opt).to(self.device)
        else:
            logger = get_root_logger()
            logger.info('No loss is registered!')

    def _setup_metrics(self):
        """
        Set up metrics for data validation
        """
        val_opt = deepcopy(self.opt['val'])
        if val_opt and 'metrics' in val_opt:
            for name, metric_opt in val_opt['metrics'].items():
                self.metrics[name] = build_metric(metric_opt)
        else:
            logger = get_root_logger()
            logger.info('No metric is registered!')

    def _setup_networks(self):
        """
        Set up networks
        """
        for name, network_opt in deepcopy(self.opt['networks']).items():
            self.networks[name] = build_network(network_opt)

    def _networks_to_device(self):
        """
        Move networks to device.
        It warps networks with DistributedDataParallel or DataParallel.
        """
        for name, network in self.networks.items():
            network = network.to(self.device)
            if self.opt['num_gpu'] > 1:
                if self.opt['backend'] == 'ddp':
                    find_unused_parameters = self.opt.get('find_unused_parameters', False)
                    network = DistributedDataParallel(
                        network, device_ids=[torch.cuda.current_device()],
                        find_unused_parameters=find_unused_parameters
                    )
                elif self.opt['backend'] == 'dp':
                    network = DataParallel(network, device_ids=list(range(self.opt['num_gpu'])))
                else:
                    raise ValueError(f'Invalid backend: {self.opt["backend"]}, only supports "dp", "ddp"')
            self.networks[name] = network

    def _get_bare_net(self, net):
        """
        Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def _get_networks_state_dict(self):
        """
        Get networks state dict.
        """
        state_dict = dict()
        for name in self.networks:
            state_dict[name] = deepcopy(self._get_bare_net(self.networks[name]).state_dict())

        return state_dict

    @master_only
    def print_networks(self):
        """
        Print the str and parameter number of networks
        """
        for net in self.networks.values():
            if isinstance(net, (DataParallel, DistributedDataParallel)):
                net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
            else:
                net_cls_str = f'{net.__class__.__name__}'

            bare_net = self._get_bare_net(net)
            net_params = sum(map(lambda x: x.numel(), bare_net.parameters()))

            logger = get_root_logger()
            logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')

    def train(self):
        """
        Setup networks to train mode
        """
        self.is_train = True
        for name in self.networks:
            self.networks[name].train()

    def eval(self):
        """
        Setup networks to eval mode
        """
        self.is_train = False
        for name in self.networks:
            self.networks[name].eval()

    @master_only
    def save_model(self, net_only=False, best=False):
        """
        Save model during training, which will be used for resuming.
        Args:
            net_only (bool): only save the network state dict. Default False.
            best (bool): save the best model state dict. Default False.
        """
        if best:
            networks_state_dict = self.best_networks_state_dict
        else:
            networks_state_dict = self._get_networks_state_dict()

        if net_only:
            state_dict = {'networks': networks_state_dict}
            save_filename = 'final.pth'
        else:
            state_dict = {
                'networks': networks_state_dict,
                'epoch': self.curr_epoch,
                'iter': self.curr_iter,
                'optimizers': {},
                'schedulers': {}
            }

            for name in self.optimizers:
                state_dict['optimizers'][name] = self.optimizers[name].state_dict()

            for name in self.schedulers:
                state_dict['schedulers'][name] = self.schedulers[name].state_dict()

            save_filename = f'{self.curr_iter}.pth'

        save_path = os.path.join(self.opt['path']['models'], save_filename)

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(state_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')

    def resume_model(self, resume_state, net_only=False, verbose=True):
        """Reload the net, optimizers and schedulers.

        Args:
            resume_state (dict): Resume state.
            net_only (bool): only resume the network state dict. Default False.
            verbose (bool): print the resuming process
        """
        networks_state_dict = resume_state['networks']

        # resume networks
        for name in self.networks:
            if len(list(self.networks[name].parameters())) == 0:
                if verbose:
                    logger = get_root_logger()
                    logger.info(f'Network {name} has no param. Ignore it.')
                continue
            if name not in networks_state_dict:
                if verbose:
                    logger = get_root_logger()
                    logger.warning(f'Network {name} cannot be resumed.')
                continue

            net_state_dict = networks_state_dict[name]
            # remove unnecessary 'module.'
            net_state_dict = {k.replace('module.', ''): v for k, v in net_state_dict.items()}

            self._get_bare_net(self.networks[name]).load_state_dict(net_state_dict)

            if verbose:
                logger = get_root_logger()
                logger.info(f"Resuming network: {name}")

        # resume optimizers and schedulers
        if not net_only:
            optimizers_state_dict = resume_state['optimizers']
            schedulers_state_dict = resume_state['schedulers']
            for name in self.optimizers:
                if name not in optimizers_state_dict:
                    if verbose:
                        logger = get_root_logger()
                        logger.warning(f'Optimizer {name} cannot be resumed.')
                    continue
                self.optimizers[name].load_state_dict(optimizers_state_dict[name])
            for name in self.schedulers:
                if name not in schedulers_state_dict:
                    if verbose:
                        logger = get_root_logger()
                        logger.warning(f'Scheduler {name} cannot be resumed.')
                    continue
                self.schedulers[name].load_state_dict(schedulers_state_dict[name])

            # resume epoch and iter
            self.curr_iter = resume_state['iter']
            self.curr_epoch = resume_state['epoch']
            if verbose:
                logger = get_root_logger()
                logger.info(f"Resuming training from epoch: {self.curr_epoch}, " f"iter: {self.curr_iter}.")

    @torch.no_grad()
    def _reduce_loss_dict(self):
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        """
        keys = []
        losses = []
        for name, value in self.loss_metrics.items():
            keys.append(name)
            losses.append(value)
        losses = torch.stack(losses, 0)
        torch.distributed.reduce(losses, dst=0)
        losses /= self.opt['world_size']
        loss_dict = {key: loss for key, loss in zip(keys, losses)}

        return loss_dict
