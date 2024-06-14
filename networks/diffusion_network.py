# The implementation of DiffusionNet is adapted from official implementation: https://github.com/nmwsharp/diffusion-net

import torch
import torch.nn as nn

from utils.geometry_util import get_all_operators, to_basis, from_basis, compute_hks_autoscale, compute_wks_autoscale, data_augmentation
from utils.registry import NETWORK_REGISTRY


class LearnedTimeDiffusion(nn.Module):
    """
    Applied diffusion with learned time per-channel.

    In the spectral domain this becomes
        f_out = e ^ (lambda_i * t) * f_in
    """
    def __init__(self, in_channels, method='spectral'):
        """
        Args:
            in_channels (int): number of input channels.
            method (str, optional): method to perform time diffusion. Default 'spectral'.
        """
        super(LearnedTimeDiffusion, self).__init__()
        assert method in ['spectral', 'implicit_dense'], f'Invalid method: {method}'
        self.in_channels = in_channels
        self.diffusion_time = nn.Parameter(torch.Tensor(in_channels))
        self.method = method
        # init as zero
        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, feat, L, mass, evals, evecs):
        """
        Args:
            feat (torch.Tensor): feature vector [B, V, C].
            L (torch.SparseTensor): sparse Laplacian matrix [B, V, V].
            mass (torch.Tensor): diagonal elements in mass matrix [B, V].
            evals (torch.Tensor): eigenvalues of Laplacian matrix [B, K].
            evecs (torch.Tensor): eigenvectors of Laplacian matrix [B, V, K].
        Returns:
            feat_diffuse (torch.Tensor): diffused feature vector [B, V, C].
        """
        # project times to the positive half-space
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        assert feat.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat.shape[-1]}'

        if self.method == 'spectral':
            # Transform to spectral
            feat_spec = to_basis(feat, evecs, mass)

            # Diffuse
            diffuse_coefs = torch.exp(-evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0))
            feat_diffuse_spec = diffuse_coefs * feat_spec

            # Transform back to feature
            feat_diffuse = from_basis(feat_diffuse_spec, evecs)

        else: # 'implicit_dense'
            V = feat.shape[-2]

            # Form the dense matrix (M + tL) with dims (B, C, V, V)
            mat_dense = L.to_dense().unsuqeeze(1).expand(-1, self.in_channels, -1, -1).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)

            # Solve the system
            rhs = feat * mass.unsqueeze(-1)
            rhsT = rhs.transpose(1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            feat_diffuse = sols.squeeze(-1).transpose(1, 2)

        return feat_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors.
    Uses a learned complex-linear layer to keep dimension down.
    """
    def __init__(self, in_channels, with_gradient_rotations=True):
        """
        Args:
            in_channels (int): number of input channels.
            with_gradient_rotations (bool, optional): whether with gradient rotations. Default True.
        """
        super(SpatialGradientFeatures, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.A_im = nn.Linear(self.in_channels, self.in_channels, bias=False)
        else:
            self.A = nn.Linear(self.in_channels, self.in_channels, bias=False)

    def forward(self, feat_in):
        """
        Args:
            feat_in (torch.Tensor): input feature vector (B, V, C, 2).
        Returns:
            feat_out (torch.Tensor): output feature vector (B, V, C)
        """
        feat_a = feat_in

        if self.with_gradient_rotations:
            feat_real_b = self.A_re(feat_in[..., 0]) - self.A_im(feat_in[..., 1])
            feat_img_b = self.A_re(feat_in[..., 0]) + self.A_im(feat_in[..., 1])
        else:
            feat_real_b = self.A(feat_in[..., 0])
            feat_img_b = self.A(feat_in[..., 1])

        feat_out = feat_a[..., 0] * feat_real_b + feat_a[..., 1] * feat_img_b

        return torch.tanh(feat_out)


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes
    """
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name='miniMLP'):
        """
        Args:
            layer_sizes (List): list of layer size.
            dropout (bool, optional): whether use dropout. Default False.
            activation (nn.Module, optional): activation function. Default ReLU.
            name (str, optional): module name. Default 'miniMLP'
        """
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            # Dropout Layer
            if dropout and i > 0:
                self.add_module(
                    name + '_dropout_{:03d}'.format(i),
                    nn.Dropout(p=0.5)
                )

            # Affine Layer
            self.add_module(
                name + '_linear_{:03d}'.format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )

            # Activation Layer
            if not is_last:
                self.add_module(
                    name + '_activation_{:03d}'.format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Building Block of DiffusionNet.
    """
    def __init__(self, in_channels, mlp_hidden_channels,
                 dropout=True,
                 diffusion_method='spectral',
                 with_gradient_features=True,
                 with_gradient_rotations=True):
        """
        Args:
            in_channels (int): number of input channels.
            mlp_hidden_channels (List): list of mlp hidden channels.
            dropout (bool, optional): whether use dropout in MLP. Default True.
            with_gradient_features (bool, optional): whether use spatial gradient feature. Default True.
            with_gradient_rotations (bool, optional): whether use spatial gradient rotation. Default True.
        """
        super(DiffusionNetBlock, self).__init__()

        self.in_channels = in_channels
        self.mlp_hidden_channels = mlp_hidden_channels
        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.in_channels, method=diffusion_method)

        # concat of both diffused features and original features
        self.mlp_in_channels = 2 * self.in_channels

        # Spatial gradient block
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.in_channels,
                                                             with_gradient_rotations=self.with_gradient_rotations)
            # concat of gradient features
            self.mlp_in_channels += self.in_channels

        # MLP block
        self.mlp = MiniMLP([self.mlp_in_channels] + self.mlp_hidden_channels + [self.in_channels], dropout=self.dropout)

    def forward(self, feat_in, mass, L, evals, evecs, gradX, gradY):
        """
        Args:
            feat_in (torch.Tensor): input feature vector [B, V, C].
            mass (torch.Tensor): diagonal elements of mass matrix [B, V].
            L (torch.SparseTensor): sparse Laplacian matrix [B, V, V].
            evals (torch.Tensor): eigenvalues of Laplacian Matrix [B, K].
            evecs (torch.Tensor): eigenvectors of Laplacian Matrix [B, V, K].
            gradX (torch.SparseTensor): real part of gradient matrix [B, V, V].
            gradY (torch.SparseTensor): imaginary part of gradient matrix [B, V, V].
        """

        B = feat_in.shape[0]
        assert feat_in.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat_in.shape[-1]}'

        # Diffusion block
        feat_diffuse = self.diffusion(feat_in, L, mass, evals, evecs)

        # Compute gradient features
        if self.with_gradient_features:
            # Compute gradient
            feat_grads = []
            for b in range(B):
                # gradient after diffusion
                feat_gradX = torch.mm(gradX[b, ...], feat_diffuse[b, ...])
                feat_gradY = torch.mm(gradY[b, ...], feat_diffuse[b, ...])

                feat_grads.append(torch.stack((feat_gradX, feat_gradY), dim=-1))
            feat_grad = torch.stack(feat_grads, dim=0) # [B, V, C, 2]

            # Compute gradient features
            feat_grad_features = self.gradient_features(feat_grad)

            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse, feat_grad_features), dim=-1)
        else:
            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse), dim=-1)

        # MLP block
        feat_out = self.mlp(feat_combined)

        # Skip connection
        feat_out = feat_out + feat_in

        return feat_out


@NETWORK_REGISTRY.register()
class DiffusionNet(nn.Module):
    """
    DiffusionNet: stacked of DiffusionBlock
    """
    def __init__(self, in_channels, out_channels,
                 hidden_channels=128,
                 n_block=4,
                 last_activation=None,
                 mlp_hidden_channels=None,
                 output_at='vertices',
                 dropout=True,
                 with_gradient_features=True,
                 with_gradient_rotations=True,
                 diffusion_method='spectral',
                 k_eig=128,
                 cache_dir=None,
                 input_type='xyz'
                 ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            hidden_channels (int, optional): number of hidden channels in diffusion block. Default 128.
            n_block (int, optional): number of diffusion blocks. Default 4.
            last_activation (nn.Module, optional): output layer. Default None.
            mlp_hidden_channels (List, optional): mlp hidden layers. Default None means [hidden_channels, hidden_channels].
            output_at (str, optional): produce outputs at various mesh elements by averaging from vertices.
            One of ['vertices', 'edges', 'faces', 'global_mean']. Default 'vertices'.
            dropout (bool, optional): whether use dropout in mlp. Default True.
            with_gradient_features (bool, optional): whether use SpatialGradientFeatures in DiffusionBlock. Default True.
            with_gradient_rotations (bool, optional): whether use gradient rotations in SpatialGradientFeatures. Default True.
            diffusion_method (str, optional): diffusion method applied in diffusion layer.
            One of ['spectral', 'implicit_dense']. Default 'spectral'.
            k_eig (int, optional): number of eigenvalues/eigenvectors to compute diffusion. Default 128.
            cache_dir (str, optional): cache dir contains all pre-computed spectral operators. Default None.
            input_type (str, optional): input type. One of ['xyz', 'shot', 'hks'] Default 'xyz'.
        """
        super(DiffusionNet, self).__init__()
        # sanity check
        assert diffusion_method in ['spectral', 'implicit_dense'], f'Invalid diffusion method: {diffusion_method}'
        assert output_at in ['vertices', 'edges', 'faces', 'global_mean'], f'Invalid output_at: {output_at}'

        # basic params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_block = n_block
        self.cache_dir = cache_dir
        self.input_type = input_type

        # output params
        self.last_activation = last_activation
        self.output_at = output_at

        # mlp options
        if not mlp_hidden_channels:
            mlp_hidden_channels = [hidden_channels, hidden_channels]
        self.mlp_hidden_channels = mlp_hidden_channels
        self.dropout = dropout

        # diffusion options
        self.diffusion_method = diffusion_method
        self.k_eig = k_eig

        # gradient feature options
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # setup networks

        # first and last linear layers
        self.first_linear = nn.Linear(in_channels, hidden_channels)
        self.last_linear = nn.Linear(hidden_channels, out_channels)

        # diffusion blocks
        blocks = []
        for i_block in range(self.n_block):
            block = DiffusionNetBlock(
                in_channels=hidden_channels,
                mlp_hidden_channels=mlp_hidden_channels,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations
            )
            blocks += [block]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, verts, faces=None, feats=None):
        assert verts.dim() == 3, 'Only support batch operation'
        if faces is not None:
            assert faces.dim() == 3, 'Only support batch operation'

        # ensure reproducibility to first convert to cpu to find the precomputed spectral ops
        if faces is not None:
            _, mass, L, evals, evecs, gradX, gradY = get_all_operators(verts.cpu(), faces.cpu(), k=self.k_eig,
                                                                       cache_dir=self.cache_dir)
        else:
            _, mass, L, evals, evecs, gradX, gradY = get_all_operators(verts.cpu(), None, k=self.k_eig,
                                                                       cache_dir=self.cache_dir)
        mass = mass.to(device=verts.device)
        L = L.to(device=verts.device)
        evals = evals.to(device=verts.device)
        evecs = evecs.to(device=verts.device)
        gradX = gradX.to(device=verts.device)
        gradY = gradY.to(device=verts.device)

        # Compute hks when necessary
        if feats is not None:
            x = feats
        else:
            if self.input_type == 'hks':
                x = compute_hks_autoscale(evals, evecs)
            elif self.input_type == 'wks':
                x = compute_wks_autoscale(evals, evecs, mass)
            elif self.input_type == 'xyz':
                if self.training:
                    verts = data_augmentation(verts)
                x = verts

        # Apply the first linear layer
        x = self.first_linear(x)

        # Apply each of the diffusion block
        for block in self.blocks:
            x = block(x, mass, L, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x = self.last_linear(x)

        # remap output to faces/edges if requested
        if self.output_at == 'vertices':
            x_out = x
        elif self.output_at == 'faces':
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            x_out = torch.gather(x_gather, 1, faces_gather).mean(dim=-1)
        else:  # global mean
            # Using a weighted mean according to the point mass/area is discretization-invariant.
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-1) / torch.sum(mass, dim=-1, keepdim=True)

        # Apply last non-linearity if specified
        if self.last_activation:
            x_out = self.last_activation(x_out)

        return x_out
