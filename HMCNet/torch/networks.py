import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import ReLU, LeakyReLU
import sys
import itertools
from functools import partial
# import pytorch_wavelets.dwt.transform3d as dwt3d
# from kymatio.torch import Scattering3D

from . import block

# sys.path.append('/home/boys/project/voxelmorph/voxelmorph_code/voxelmorph/torch')
# from typing import Optional
# from torch import nn, Tensor
# from torch.nn.init import trunc_normal_

# # from .context import is_fna_enabled
# # from .experimental import na3d as experimental_na3d
# from .functional import na3d, na3d_av, na3d_qk
# from .types import CausalArg3DTypeOrDed, Dimension3DTypeOrDed

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args
# from patchify import patchify
import random
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
import torch.utils.checkpoint as checkpoint
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import einops
# from mamba_ssm import Mamba
from timm.models.swin_transformer import swin_large_patch4_window7_224,swin_large_patch4_window12_384


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel_size=3,stride=1,padding = 1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding=padding)
        # self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        # out = self.norm(out)
        out = self.activation(out)
        return out


def compute_mask(dims, window_size, shift_size, device): #([80,100,115]),([5,5,5]) ([2,2,2])
    
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition(x_in, window_size):

    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        
    return windows


def window_reverse(windows, window_size, dims):

    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    
    
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b) 
        return tensor
    

    
    
class PatchExpanding_block(nn.Module):
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.up_conv = nn.ConvTranspose3d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim//2)

    def forward(self, x_in):

        x = self.up_conv(x_in)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')
        
        return x_out
    

class SwinTrans_stage_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
        
        self.window_size = window_size  # 5
        self.shift_size = tuple(i // 2 for i in window_size)  # 2
        self.no_shift = tuple(0 for i in window_size)  # 0
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    window_size=self.window_size,
                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    use_checkpoint=use_checkpoint)
            self.blocks.append(block)
        
    def forward(self, x_in):
        
        b, c, d, h, w = x_in.shape  # 1,16,80,96,112
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)  # ([5,5,5]) ([2,2,2])
        dp = int(np.ceil(d / window_size[0])) * window_size[0] # 80
        hp = int(np.ceil(h / window_size[1])) * window_size[1] # 100
        wp = int(np.ceil(w / window_size[2])) * window_size[2] # 115
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)
        
        x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out
    

class SwinTrans_Block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 shift_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
                         
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
                         
        self.norm1 = nn.LayerNorm(embed_dim)  
        self.attn = MSA_block(embed_dim,
                              window_size=window_size,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(hidden_size=embed_dim, 
                             mlp_dim=int(embed_dim * mlp_ratio), 
                             dropout_rate=drop)

    def forward_part1(self, x_in, mask_matrix):
        
        x = self.norm1(x_in)
        
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None  
         
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        
        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):
        
        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, mask_matrix=None):
        
        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)
                         
        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x_out = x + self.forward_part2(x)
        
        return x_out


class MSA_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * 
                                                                     (2 * self.window_size[1] - 1) * 
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, mask=None):
        
        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x_out = self.proj_drop(x)
        
        return x_out
    
    
class MLP_block(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.GELU = nn.GELU()

    def forward(self, x_in):
        
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)
        
        x = self.linear2(x)
        x_out = self.drop2(x)
        
        return x_out


class SCA3D(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel//2, int(channel // (2*reduction))),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // (2*reduction)), channel//2))
        # self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1,stride=1, padding=0, bias=False)
        self.spatial_se = nn.Conv3d(channel//2, 1,  kernel_size=(1,1,1), stride=1, padding=0, bias=False)

    def channel_shuffle(self,x, groups):

        b, c, h, w, d = x.shape

        x = x.reshape(b, groups, -1, h, w, d)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # flatten
        x = x.reshape(b, -1, h, w, d)

        return x

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()
        x = self.channel_shuffle(x,2)
        x_0, x_1 = x.chunk(2, dim=1)
        chn_se = self.avg_pool(x_0).view(bahs, chs//2)
        # print(chn_se.shape)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs//2, 1, 1, 1))
        chn_se = torch.mul(x_0, chn_se)
        spa_se = torch.sigmoid(self.spatial_se(x_1))
        spa_se = torch.mul(x_1, spa_se)
        out = torch.cat([chn_se, spa_se], dim=1)
        # net_out = spa_se + x + chn_se
        out = out + x
        # print(out.shape)
        # out = out.reshape(bahs, -1, _, _, _)
        # out = self.channel_shuffle(out,2)
        return out



class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)  # 3D 适用于体数据（如 MRI）
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # 跳跃连接层（恒等映射）
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        identity = self.shortcut(x)  # 记录跳跃连接

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out += identity  # 加上跳跃连接
        return out

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        self.transformer1 = layers.SpatialTransformer((24,20,24))
        self.transformer2 = layers.SpatialTransformer((48,40,48))
        self.transformer3 = layers.SpatialTransformer((96,80,96))
        self.transformer4 = layers.SpatialTransformer((192,160,192))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))

        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))


        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,16)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,16,32)
        self.encoder3_m = ConvBlock(3,32,32)
        self.encoder4_m = ConvBlock(3,32,32)


        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m

        
        self.decoder1 = ConvBlock(ndims,64,32)
        self.decoder2 = ConvBlock(ndims,96,32)
        self.decoder2_iterative_1 = ConvBlock(ndims,96,32)
        self.decoder3 = ConvBlock(ndims,96,32)
        self.decoder3_iterative_1 = ConvBlock(ndims,96,32)
        self.decoder4 = ConvBlock(ndims,96,16)
        self.decoder4_iterative_1 = ConvBlock(ndims,80,16)
        self.decoder5 = ConvBlock(ndims,48,16)

        self.trans_2 = SwinTrans_stage_block(embed_dim=32,    # 16
                                             num_layers=2,
                                             num_heads=2,    # 1
                                             window_size=[5,5,5],
                                             use_checkpoint=False)

        
        self.trans_3 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_3_2 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[7,7,7],
                                             use_checkpoint=False)
        self.trans_3_3 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[3,3,3],
                                             use_checkpoint=False)
        self.trans_conv_1 = ConvBlock(ndims,96,32)

        self.trans_3_iterative_1 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_3_2_iterative_1 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[7,7,7],
                                             use_checkpoint=False)
        self.trans_3_3_iterative_1 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[3,3,3],
                                             use_checkpoint=False)
        self.trans_conv_1_iterative_1 = ConvBlock(ndims,96,32)


        self.trans_4 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_4_2 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[3,3,3],
                                             use_checkpoint=False)
        self.trans_4_3 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[7,7,7],
                                             use_checkpoint=False)
        self.trans_conv_2 = ConvBlock(ndims,96,32)
        
        
        self.trans_4_iterative_1 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_4_2_iterative_1 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[3,3,3],
                                             use_checkpoint=False)
        self.trans_4_3_iterative_1 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[7,7,7],
                                             use_checkpoint=False)
        self.trans_conv_2_iterative_1 = ConvBlock(ndims,96,32)

      
        self.trans_5 = SwinTrans_stage_block(embed_dim=16, # 128
                                             num_layers=2,
                                             num_heads=2, # 8
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_5_2 = SwinTrans_stage_block(embed_dim=16, # 128
                                             num_layers=2,
                                             num_heads=2, # 8
                                             window_size=[3,3,3],
                                             use_checkpoint=False)
        self.trans_conv_3 = ConvBlock(ndims,32,16)

        self.trans_5_iterative_1 = SwinTrans_stage_block(embed_dim=16, # 128
                                             num_layers=2,
                                             num_heads=2, # 8
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_5_2_iterative_1 = SwinTrans_stage_block(embed_dim=16, # 128
                                             num_layers=2,
                                             num_heads=2, # 8
                                             window_size=[3,3,3],
                                             use_checkpoint=False)
        self.trans_conv_3_iterative_1 = ConvBlock(ndims,32,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1_iterative_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2_iterative_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))
        self.output_block_3_iterative_1 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f1_scale4 = self.resblock_seq(16,16)
        self.fusion_f2_scale4 = self.resblock_seq(32,32)
        self.fusion_f3_scale4 = self.resblock_seq(32,32)
        self.fusion_f4_scale4 = self.resblock_seq(32,32)
        self.fusion_f_scale4 = self.resblock_seq(112,32)

        self.CBAM_f1_scale4 = SCA3D(16)
        self.CBAM_f2_scale4 = SCA3D(32)
        self.CBAM_f3_scale4 = SCA3D(32)
        self.CBAM_f4_scale4 = SCA3D(32)
        self.CBAM_f_scale4 = SCA3D(112)

        self.fusion_m1_scale4 = self.fusion_f1_scale4
        self.fusion_m2_scale4 = self.fusion_f2_scale4
        self.fusion_m3_scale4 = self.fusion_f3_scale4
        self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4

        self.CBAM_m1_scale4 = self.CBAM_f1_scale4
        self.CBAM_m2_scale4 = self.CBAM_f2_scale4
        self.CBAM_m3_scale4 = self.CBAM_f3_scale4
        self.CBAM_m4_scale4 = self.CBAM_f4_scale4
        self.CBAM_m_scale4 = self.CBAM_f_scale4

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f1_scale3 = self.resblock_seq(16,16)
        self.fusion_f2_scale3 = self.resblock_seq(32,32)
        self.fusion_f3_scale3 = self.resblock_seq(32,32)
        self.fusion_f4_scale3 = self.resblock_seq(32,32)
        self.fusion_f_scale3 = self.resblock_seq(112,32)

        self.CBAM_f1_scale3 = SCA3D(16)
        self.CBAM_f2_scale3 = SCA3D(32)
        self.CBAM_f3_scale3 = SCA3D(32)
        self.CBAM_f4_scale3 = SCA3D(32)
        self.CBAM_f_scale3 = SCA3D(112)

        self.fusion_m1_scale3 = self.fusion_f1_scale3
        self.fusion_m2_scale3 = self.fusion_f2_scale3
        self.fusion_m3_scale3 = self.fusion_f3_scale3
        self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3

        self.CBAM_m1_scale3 = self.CBAM_f1_scale3
        self.CBAM_m2_scale3 = self.CBAM_f2_scale3
        self.CBAM_m3_scale3 = self.CBAM_f3_scale3
        self.CBAM_m4_scale3 = self.CBAM_f4_scale3
        self.CBAM_m_scale3 = self.CBAM_f_scale3

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f1_scale2 = self.resblock_seq(16,16)
        self.fusion_f2_scale2 = self.resblock_seq(32,32)
        self.fusion_f3_scale2 = self.resblock_seq(32,32)
        self.fusion_f4_scale2 = self.resblock_seq(32,32)
        self.fusion_f_scale2 = self.resblock_seq(112,32)

        self.CBAM_f1_scale2 = SCA3D(16)
        self.CBAM_f2_scale2 = SCA3D(32)
        self.CBAM_f3_scale2 = SCA3D(32)
        self.CBAM_f4_scale2 = SCA3D(32)
        self.CBAM_f_scale2 = SCA3D(112)

        self.fusion_m1_scale2 = self.fusion_f1_scale2
        self.fusion_m2_scale2 = self.fusion_f2_scale2
        self.fusion_m3_scale2 = self.fusion_f3_scale2
        self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2

        self.CBAM_m1_scale2 = self.CBAM_f1_scale2
        self.CBAM_m2_scale2 = self.CBAM_f2_scale2
        self.CBAM_m3_scale2 = self.CBAM_f3_scale2
        self.CBAM_m4_scale2 = self.CBAM_f4_scale2
        self.CBAM_m_scale2 = self.CBAM_f_scale2



    def resblock_seq(self, in_channels, out_channels, bias_opt=False):
        '''
        一个resblock_seq作为一个编码器, 不需要太多,
        借鉴LapIRN
        '''
        layer = nn.Sequential(
            PreActBlock(in_channels, out_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer
    
    
    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,16,160,192,224)
        xf_skip_1 = self.encoder1_f(xf) # (1,16,160,192,224)

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,16,80,96,112)
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,16,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)




        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        # 第一个decoder
        x = torch.cat([xm, xf], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)
        x = self.trans_2(x)

        # ------------------------------Scale4--------------------------------------------
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        xm_skip_1_fusion4 = self.CBAM_m1_scale4(xm_skip_1_fusion4)
        xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 

        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,16,20,24,28)
        xm_skip_2_fusion4 = self.CBAM_m2_scale4(xm_skip_2_fusion4)
        xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)

        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        xm_skip_3_fusion4 = self.CBAM_m3_scale4(xm_skip_3_fusion4)
        xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)

        xm_skip_4 = self.CBAM_m4_scale4(xm_skip_4)
        xm_skip_4_fusion4 = self.fusion_m4_scale4(xm_skip_4) # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)


        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        xf_skip_1_fusion4 = self.CBAM_f1_scale4(xf_skip_1_fusion4)
        xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)

        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,16,20,24,28)
        xf_skip_2_fusion4 = self.CBAM_f2_scale4(xf_skip_2_fusion4)
        xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)

        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        xf_skip_3_fusion4 = self.CBAM_f3_scale4(xf_skip_3_fusion4)
        xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)

        xf_skip_4 = self.CBAM_f4_scale4(xf_skip_4)
        xf_skip_4_fusion4 = self.fusion_f4_scale4(xf_skip_4) # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4) # (1,32,20,24,28)

        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        x_1 = self.trans_3(x)
        x_2 = self.trans_3_2(x)
        x_3 = self.trans_3_3(x)
        x_1 = torch.cat([x_1, x_2], dim=1)
        x = self.trans_conv_1(torch.cat([x_1, x_3], dim=1))

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        

        # 迭代
        xw_fusion4 = self.transformer1(xm_fusion4, flow_1)
        x_fusion4_iterative_1 = torch.cat([xw_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)
        x = self.decoder2_iterative_1(torch.cat([x, x_fusion4_iterative_1], dim=1)) # (1,32,20,24,28)
        x_1 = self.trans_3_iterative_1(x)
        x_2 = self.trans_3_2_iterative_1(x)
        x_3 = self.trans_3_3_iterative_1(x)
        x_1 = torch.cat([x_1, x_2], dim=1)
        x = self.trans_conv_1_iterative_1(torch.cat([x_1, x_3], dim=1))

        delta_flow_1 = self.output_block_1_iterative_1(x) # (1,3,20,24,28)
        flow_1 = self.transformer1(flow_1, delta_flow_1) + delta_flow_1 

        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2, mode="trilinear")*2 # (1,3,40,48,56)




        ## ------------------------------Scale3--------------------------------------------
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        xm_skip_1_fusion3 = self.CBAM_m1_scale3(xm_skip_1_fusion3)
        xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 

        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,16,40,48,56)
        xm_skip_2_fusion3 = self.CBAM_m2_scale3(xm_skip_2_fusion3)
        xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)

        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        xm_skip_3_fusion3 = self.CBAM_m3_scale3(xm_skip_3_fusion3)
        xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        xm_skip_4 = self.CBAM_m4_scale3(xm_skip_4)
        xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4_fusion3, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,88,40,48,56)

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)

        xw_fusion3 = self.fusion_m_scale3(xw_fusion3) # (1,32,40,48,56)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        xf_skip_1_fusion3 = self.CBAM_f1_scale3(xf_skip_1_fusion3)
        xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 

        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,16,40,48,56)
        xf_skip_2_fusion3 = self.CBAM_f2_scale3(xf_skip_2_fusion3)
        xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)

        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)
        xf_skip_3_fusion3 = self.CBAM_f3_scale3(xf_skip_3_fusion3)
        xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        xf_skip_4 = self.CBAM_f4_scale3(xf_skip_4)
        xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4_fusion3, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,88,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)
        x_1 = self.trans_4(x)
        x_2 = self.trans_4_2(x)
        x_3 = self.trans_4_3(x)
        x_1 = torch.cat([x_1, x_2], dim=1)
        x = self.trans_conv_2(torch.cat([x_1, x_3], dim=1))

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = self.transformer2(flow_1_up, delta_flow_2)+delta_flow_2
        # flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        # ------- 迭代 ---------
        xw_fusion3 = self.transformer2(xw_fusion3, delta_flow_2)
        x_fusion3_iterative_1 = torch.cat([xw_fusion3, xf_fusion3], dim=1) # (1,64,20,24,28)
        x = self.decoder3_iterative_1(torch.cat([x, x_fusion3_iterative_1], dim=1)) # (1,32,20,24,28)
        # x = self.trans_4_iterative_1(x)
        x_1 = self.trans_4_iterative_1(x)
        x_2 = self.trans_4_2_iterative_1(x)
        x_3 = self.trans_4_3_iterative_1(x)
        x_1 = torch.cat([x_1, x_2], dim=1)
        x = self.trans_conv_2_iterative_1(torch.cat([x_1, x_3], dim=1))

        delta_flow_2 = self.output_block_2_iterative_1(x) # (1,3,20,24,28)
        flow_2 = self.transformer2(flow_2, delta_flow_2) + delta_flow_2 

        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2, mode="trilinear")*2 # (1,3,40,48,56)



        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_1_fusion2 = self.CBAM_m1_scale2(xm_skip_1_fusion2)
        xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 


        xm_skip_2_fusion2 = xm_skip_2 # (1,16,80,96,112)
        xm_skip_2_fusion2 = self.CBAM_m2_scale2(xm_skip_2_fusion2)
        xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        xm_skip_3 = self.CBAM_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3_fusion2, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xm_skip_4 = self.CBAM_m4_scale2(xm_skip_4)
        xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,32,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4_fusion2, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,96,80,96,112)

        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xw_fusion2 = self.fusion_m_scale2(xw_fusion2) # (1,32,80,96,112)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        xf_skip_1_fusion2 = self.CBAM_f1_scale2(xf_skip_1_fusion2)
        xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 

        xf_skip_2_fusion2 = xf_skip_2 # (1,16,80,96,112)
        xf_skip_2_fusion2 = self.CBAM_f2_scale2(xf_skip_2_fusion2)
        xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        xf_skip_3 = self.CBAM_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3_fusion2, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xf_skip_4 = self.CBAM_f4_scale2(xf_skip_4)
        xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4_fusion2, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,16,80,96,112)

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        x_1 = self.trans_5(x)
        x_2 = self.trans_5_2(x)
        x = self.trans_conv_3(torch.cat([x_1, x_2], dim=1))
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = self.transformer3(flow_2_up, delta_flow_3) + delta_flow_3
        # flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # 迭代
        xw_fusion2 = self.transformer3(xw_fusion2, delta_flow_3)
        x_fusion2_iterative_1 = torch.cat([xw_fusion2, xf_fusion2], dim=1) # (1,64,20,24,28)
        x = self.decoder4_iterative_1(torch.cat([x, x_fusion2_iterative_1], dim=1)) # (1,32,20,24,28)
        # x = self.trans_5_iterative_1(x)
        x_1 = self.trans_5_iterative_1(x)
        x_2 = self.trans_5_2_iterative_1(x)
        x = self.trans_conv_3_iterative_1(torch.cat([x_1, x_2], dim=1))

        delta_flow_3 = self.output_block_3_iterative_1(x) # (1,3,20,24,28)
        flow_3 = self.transformer3(flow_3, delta_flow_3) + delta_flow_3 

        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear")*2 # (1,3,40,48,56)


        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)        
        flow_final = self.transformer4(flow_3_up, delta_flow_final)+delta_flow_final
            

        return flow_1,flow_2,flow_3,flow_final


class dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_diff(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    两个权重共享的编码器来各自提取特征, 然后特征融合, 送入同一个解码器。
    集成三个并行的不同窗口大小的 Swin Transformer 块，并引入 VecInt 微分同胚流场积分机制。
    """

    @store_config_args
    def __init__(self, inshape=(160, 192, 224)):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # ------------------------------微分同胚核心组件------------------------------------
        # 动态匹配多尺度金字塔的真实空间尺寸，避免硬编码 (16, 16, 16) 导致的尺寸不匹配问题
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            # i=0: Scale1 (160, 192, 224)
            # i=1: Scale2 (80, 96, 112)
            # i=2: Scale3 (40, 48, 56)
            # i=3: Scale4 (20, 24, 28)
            current_shape = [s // (2**i) for s in inshape]
            self.warp.append(SpatialTransformer(current_shape))
            self.diff.append(VecInt(current_shape)) # int_steps=7 为经典的积分步数

        # 传统的通用 STN 变换器（用于特征图的 warp 操作）
        self.transformer1 = self.warp[3] # Scale4: (20, 24, 28)
        self.transformer2 = self.warp[2] # Scale3: (40, 48, 56)
        self.transformer3 = self.warp[1] # Scale2: (80, 96, 112)
        self.transformer4 = self.warp[0] # Scale1: (160, 192, 224)
        # ---------------------------------------------------------------------------------

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 = MaxPooling(4)
        self.pooling_8 = MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 

        # 编码器 (权重共享)
        self.encoder1_m = ConvBlock(3, 1, 16)  
        self.encoder2_m = ConvBlock(3, 16, 32)
        self.encoder3_m = ConvBlock(3, 32, 32)
        self.encoder4_m = ConvBlock(3, 32, 32)

        self.encoder1_f = self.encoder1_m  
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        
        # 解码器核心块
        self.decoder1 = ConvBlock(ndims, 64, 32)
        self.decoder2 = ConvBlock(ndims, 96, 32)
        self.decoder2_iterative_1 = ConvBlock(ndims, 96, 32)
        self.decoder3 = ConvBlock(ndims, 96, 32)
        self.decoder3_iterative_1 = ConvBlock(ndims, 96, 32)
        self.decoder4 = ConvBlock(ndims, 96, 16)
        self.decoder4_iterative_1 = ConvBlock(ndims, 80, 16)
        self.decoder5 = ConvBlock(ndims, 48, 16)

        # Triple Swin Transformer 架构配置
        self.trans_2 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[5,5,5], use_checkpoint=False)

        # Scale 4
        self.trans_3 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[5,5,5], use_checkpoint=False)
        self.trans_3_2 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[7,7,7], use_checkpoint=False)
        self.trans_3_3 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[3,3,3], use_checkpoint=False)
        self.trans_conv_1 = ConvBlock(ndims, 96, 32)

        self.trans_3_iterative_1 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[5,5,5], use_checkpoint=False)
        self.trans_3_2_iterative_1 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[7,7,7], use_checkpoint=False)
        self.trans_3_3_iterative_1 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[3,3,3], use_checkpoint=False)
        self.trans_conv_1_iterative_1 = ConvBlock(ndims, 96, 32)

        # Scale 3
        self.trans_4 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[5,5,5], use_checkpoint=False)
        self.trans_4_2 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[3,3,3], use_checkpoint=False)
        self.trans_4_3 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[7,7,7], use_checkpoint=False)
        self.trans_conv_2 = ConvBlock(ndims, 96, 32)
        
        self.trans_4_iterative_1 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[5,5,5], use_checkpoint=False)
        self.trans_4_2_iterative_1 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[3,3,3], use_checkpoint=False)
        self.trans_4_3_iterative_1 = SwinTrans_stage_block(embed_dim=32, num_layers=2, num_heads=2, window_size=[7,7,7], use_checkpoint=False)
        self.trans_conv_2_iterative_1 = ConvBlock(ndims, 96, 32)

        # Scale 2
        self.trans_5 = SwinTrans_stage_block(embed_dim=16, num_layers=2, num_heads=2, window_size=[5,5,5], use_checkpoint=False)
        self.trans_5_2 = SwinTrans_stage_block(embed_dim=16, num_layers=2, num_heads=2, window_size=[3,3,3], use_checkpoint=False)
        self.trans_conv_3 = ConvBlock(ndims, 32, 16)

        self.trans_5_iterative_1 = SwinTrans_stage_block(embed_dim=16, num_layers=2, num_heads=2, window_size=[5,5,5], use_checkpoint=False)
        self.trans_5_2_iterative_1 = SwinTrans_stage_block(embed_dim=16, num_layers=2, num_heads=2, window_size=[3,3,3], use_checkpoint=False)
        self.trans_conv_3_iterative_1 = ConvBlock(ndims, 32, 16)

        # 输出卷积映射
        self.output_block = nn.Sequential(ConvBlock(ndims, 16, 16), ConvBlock(ndims, 16, 16))
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        self.output_block_1 = nn.Sequential(ConvBlock(ndims, 32, 16), ConvBlock(ndims, 16, 3))
        self.output_block_1_iterative_1 = nn.Sequential(ConvBlock(ndims, 32, 16), ConvBlock(ndims, 16, 3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims, 32, 16), ConvBlock(ndims, 16, 3))
        self.output_block_2_iterative_1 = nn.Sequential(ConvBlock(ndims, 32, 16), ConvBlock(ndims, 16, 3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims, 16, 16), ConvBlock(ndims, 16, 3))
        self.output_block_3_iterative_1 = nn.Sequential(ConvBlock(ndims, 16, 16), ConvBlock(ndims, 16, 3))

        # 特征整合架构定义 (Scale 4)
        self.fusion_f1_scale4 = self.resblock_seq(16, 16)
        self.fusion_f2_scale4 = self.resblock_seq(32, 32)
        self.fusion_f3_scale4 = self.resblock_seq(32, 32)
        self.fusion_f4_scale4 = self.resblock_seq(32, 32)
        self.fusion_f_scale4 = self.resblock_seq(112, 32)

        self.CBAM_f1_scale4 = SCA3D(16)
        self.CBAM_f2_scale4 = SCA3D(32)
        self.CBAM_f3_scale4 = SCA3D(32)
        self.CBAM_f4_scale4 = SCA3D(32)
        self.CBAM_f_scale4 = SCA3D(112)

        self.fusion_m1_scale4 = self.fusion_f1_scale4
        self.fusion_m2_scale4 = self.fusion_f2_scale4
        self.fusion_m3_scale4 = self.fusion_f3_scale4
        self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4

        self.CBAM_m1_scale4 = self.CBAM_f1_scale4
        self.CBAM_m2_scale4 = self.CBAM_f2_scale4
        self.CBAM_m3_scale4 = self.CBAM_f3_scale4
        self.CBAM_m4_scale4 = self.CBAM_f4_scale4
        self.CBAM_m_scale4 = self.CBAM_f_scale4

        # 特征整合架构定义 (Scale 3)
        self.fusion_f1_scale3 = self.resblock_seq(16, 16)
        self.fusion_f2_scale3 = self.resblock_seq(32, 32)
        self.fusion_f3_scale3 = self.resblock_seq(32, 32)
        self.fusion_f4_scale3 = self.resblock_seq(32, 32)
        self.fusion_f_scale3 = self.resblock_seq(112, 32)

        self.CBAM_f1_scale3 = SCA3D(16)
        self.CBAM_f2_scale3 = SCA3D(32)
        self.CBAM_f3_scale3 = SCA3D(32)
        self.CBAM_f4_scale3 = SCA3D(32)
        self.CBAM_f_scale3 = SCA3D(112)

        self.fusion_m1_scale3 = self.fusion_f1_scale3
        self.fusion_m2_scale3 = self.fusion_f2_scale3
        self.fusion_m3_scale3 = self.fusion_f3_scale3
        self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3

        self.CBAM_m1_scale3 = self.CBAM_f1_scale3
        self.CBAM_m2_scale3 = self.CBAM_f2_scale3
        self.CBAM_m3_scale3 = self.CBAM_f3_scale3
        self.CBAM_m4_scale3 = self.CBAM_f4_scale3
        self.CBAM_m_scale3 = self.CBAM_f_scale3

        # 特征整合架构定义 (Scale 2)
        self.fusion_f1_scale2 = self.resblock_seq(16, 16)
        self.fusion_f2_scale2 = self.resblock_seq(32, 32)
        self.fusion_f3_scale2 = self.resblock_seq(32, 32)
        self.fusion_f4_scale2 = self.resblock_seq(32, 32)
        self.fusion_f_scale2 = self.resblock_seq(112, 32)

        self.CBAM_f1_scale2 = SCA3D(16)
        self.CBAM_f2_scale2 = SCA3D(32)
        self.CBAM_f3_scale2 = SCA3D(32)
        self.CBAM_f4_scale2 = SCA3D(32)
        self.CBAM_f_scale2 = SCA3D(112)

        self.fusion_m1_scale2 = self.fusion_f1_scale2
        self.fusion_m2_scale2 = self.fusion_f2_scale2
        self.fusion_m3_scale2 = self.fusion_f3_scale2
        self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2

        self.CBAM_m1_scale2 = self.CBAM_f1_scale2
        self.CBAM_m2_scale2 = self.CBAM_f2_scale2
        self.CBAM_m3_scale2 = self.CBAM_f3_scale2
        self.CBAM_m4_scale2 = self.CBAM_f4_scale2
        self.CBAM_m_scale2 = self.CBAM_f_scale2

    def resblock_seq(self, in_channels, out_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, out_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer

    def forward(self, source, target):
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm)  # (1,16,160,192,224)
        xf_skip_1 = self.encoder1_f(xf)  # (1,16,160,192,224)

        # moving encoder 下采样
        xm = self.pooling(xm_skip_1)     # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm)  # (1,16,80,96,112)
        xm = self.pooling(xm_skip_2)     # (1,16,40,48,56)
        xm_skip_3 = self.encoder3_m(xm)  # (1,32,40,48,56)
        xm = self.pooling(xm_skip_3)     # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm)  # (1,32,20,24,28)
        xm = self.pooling(xm_skip_4)     # (1,32,10,12,14)

        # fixed encoder 下采样
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        xf = self.pooling(xf_skip_4)

        # 第一个最底层 decoder
        x = torch.cat([xm, xf], dim=1)   # (1,64,10,12,14)
        x = self.decoder1(x)             # (1,32,10,12,14)
        x = self.trans_2(x)

        # ------------------------------Scale4 (1/8分辨率: 20x24x28)-------------------------
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) 
        xm_skip_1_fusion4 = self.CBAM_m1_scale4(xm_skip_1_fusion4)
        xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 

        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) 
        xm_skip_2_fusion4 = self.CBAM_m2_scale4(xm_skip_2_fusion4)
        xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)

        xm_skip_3_fusion4 = self.pooling(xm_skip_3) 
        xm_skip_3_fusion4 = self.CBAM_m3_scale4(xm_skip_3_fusion4)
        xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)

        xm_skip_4_temp = self.CBAM_m4_scale4(xm_skip_4)
        xm_skip_4_fusion4 = self.fusion_m4_scale4(xm_skip_4_temp) 

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4, xm_skip_3_fusion4, xm_skip_4_fusion4], dim=1)
        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) 

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) 
        xf_skip_1_fusion4 = self.CBAM_f1_scale4(xf_skip_1_fusion4)
        xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)

        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) 
        xf_skip_2_fusion4 = self.CBAM_f2_scale4(xf_skip_2_fusion4)
        xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)

        xf_skip_3_fusion4 = self.pooling(xf_skip_3) 
        xf_skip_3_fusion4 = self.CBAM_f3_scale4(xf_skip_3_fusion4)
        xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)

        xf_skip_4_temp = self.CBAM_f4_scale4(xf_skip_4)
        xf_skip_4_fusion4 = self.fusion_f4_scale4(xf_skip_4_temp) 

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4, xf_skip_3_fusion4, xf_skip_4_fusion4], dim=1)
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4) 

        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) 

        # 第二个decoder & Triple Trans
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) 
        x_1 = self.trans_3(x)
        x_2 = self.trans_3_2(x)
        x_3 = self.trans_3_3(x)
        x = self.trans_conv_1(torch.cat([x_1, x_2, x_3], dim=1))

        # ----【微分同胚重构】输出1/8分辨率的初始流场-----
        velocity_1 = self.output_block_1(x) 
        flow_1 = self.diff[3](velocity_1)    # 积分变换得到微分同胚流场 flow_1

        # 迭代残差修正
        xw_fusion4 = self.transformer1(xm_fusion4, flow_1)
        x_fusion4_iterative_1 = torch.cat([xw_fusion4, xf_fusion4], dim=1) 
        x = self.decoder2_iterative_1(torch.cat([x, x_fusion4_iterative_1], dim=1)) 
        x_1 = self.trans_3_iterative_1(x)
        x_2 = self.trans_3_2_iterative_1(x)
        x_3 = self.trans_3_3_iterative_1(x)
        x = self.trans_conv_1_iterative_1(torch.cat([x_1, x_2, x_3], dim=1))

        # ----【微分同胚重构】计算残差速度场并进行空间复合-----
        delta_velocity_1 = self.output_block_1_iterative_1(x)
        delta_flow_1 = self.diff[3](delta_velocity_1) # 转化为微分同胚残差场
        flow_1 = self.warp[3](flow_1, delta_flow_1) + delta_flow_1 # 严格流场变形复合

        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2, mode="trilinear") * 2 


        ## ------------------------------Scale3 (1/4分辨率: 40x48x56)-------------------------
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) 
        xm_skip_1_fusion3 = self.CBAM_m1_scale3(xm_skip_1_fusion3)
        xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 

        xm_skip_2_fusion3 = self.pooling(xm_skip_2) 
        xm_skip_2_fusion3 = self.CBAM_m2_scale3(xm_skip_2_fusion3)
        xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)

        xm_skip_3_fusion3 = self.CBAM_m3_scale3(xm_skip_3)
        xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        xm_skip_4_temp = self.CBAM_m4_scale3(xm_skip_4)
        xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4_temp) 
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4_fusion3, scale_factor=2, mode="trilinear")

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3, xm_skip_3_fusion3, xm_skip_4_fusion3], dim=1)
        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)
        xw_fusion3 = self.fusion_m_scale3(xw_fusion3) 

        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) 
        xf_skip_1_fusion3 = self.CBAM_f1_scale3(xf_skip_1_fusion3)
        xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 

        xf_skip_2_fusion3 = self.pooling(xf_skip_2) 
        xf_skip_2_fusion3 = self.CBAM_f2_scale3(xf_skip_2_fusion3)
        xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)

        xf_skip_3_fusion3 = self.CBAM_f3_scale3(xf_skip_3)
        xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        xf_skip_4_temp = self.CBAM_f4_scale3(xf_skip_4)
        xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4_temp) 
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4_fusion3, scale_factor=2, mode="trilinear")

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3, xf_skip_3_fusion3, xf_skip_4_fusion3], dim=1)
        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) 

        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1) 

        # 第三个decoder & Triple Trans
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) 
        x_1 = self.trans_4(x)
        x_2 = self.trans_4_2(x)
        x_3 = self.trans_4_3(x)
        x = self.trans_conv_2(torch.cat([x_1, x_2, x_3], dim=1))

        # ----【微分同胚重构】计算当前层级残差并进行空间流场复合-----
        delta_velocity_2 = self.output_block_2(x) 
        delta_flow_2 = self.diff[2](delta_velocity_2)
        flow_2 = self.warp[2](flow_1_up, delta_flow_2) + delta_flow_2

        # 迭代残差修正
        xw_fusion3 = self.transformer2(xw_fusion3, delta_flow_2)
        x_fusion3_iterative_1 = torch.cat([xw_fusion3, xf_fusion3], dim=1) 
        x = self.decoder3_iterative_1(torch.cat([x, x_fusion3_iterative_1], dim=1)) 
        x_1 = self.trans_4_iterative_1(x)
        x_2 = self.trans_4_2_iterative_1(x)
        x_3 = self.trans_4_3_iterative_1(x)
        x = self.trans_conv_2_iterative_1(torch.cat([x_1, x_2, x_3], dim=1))

        # ----【微分同胚重构】计算二次迭代残差并进行空间复合-----
        delta_velocity_2_iter = self.output_block_2_iterative_1(x)
        delta_flow_2_iter = self.diff[2](delta_velocity_2_iter)
        flow_2 = self.warp[2](flow_2, delta_flow_2_iter) + delta_flow_2_iter 

        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2, mode="trilinear") * 2 


        ## ------------------------------Scale2 (1/2分辨率: 80x96x112)-------------------------
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) 
        xm_skip_1_fusion2 = self.CBAM_m1_scale2(xm_skip_1_fusion2)
        xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 

        xm_skip_2_fusion2 = self.CBAM_m2_scale2(xm_skip_2)
        xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        xm_skip_3_temp = self.CBAM_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3_temp)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3_fusion2, scale_factor=2, mode="trilinear")  

        xm_skip_4_temp = self.CBAM_m4_scale2(xm_skip_4)
        xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4_temp) 
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4_fusion2, scale_factor=4, mode="trilinear")

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2, xm_skip_3_fusion2, xm_skip_4_fusion2], dim=1)
        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)
        xw_fusion2 = self.fusion_m_scale2(xw_fusion2) 

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) 
        xf_skip_1_fusion2 = self.CBAM_f1_scale2(xf_skip_1_fusion2)
        xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 

        xf_skip_2_fusion2 = self.CBAM_f2_scale2(xf_skip_2)
        xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        xf_skip_3_temp = self.CBAM_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3_temp)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3_fusion2, scale_factor=2, mode="trilinear")  

        xf_skip_4_temp = self.CBAM_f4_scale2(xf_skip_4)
        xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4_temp) 
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4_fusion2, scale_factor=4, mode="trilinear")

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2, xf_skip_3_fusion2, xf_skip_4_fusion2], dim=1)
        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) 

        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  

        # 第四个decoder & Double Trans
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) 
        x_1 = self.trans_5(x)
        x_2 = self.trans_5_2(x)
        x = self.trans_conv_3(torch.cat([x_1, x_2], dim=1))

        # ----【微分同胚重构】计算当前层级残差并进行空间流场复合-----
        delta_velocity_3 = self.output_block_3(x) 
        delta_flow_3 = self.diff[1](delta_velocity_3)
        flow_3 = self.warp[1](flow_2_up, delta_flow_3) + delta_flow_3

        # 迭代残差修正
        xw_fusion2 = self.transformer3(xw_fusion2, delta_flow_3)
        x_fusion2_iterative_1 = torch.cat([xw_fusion2, xf_fusion2], dim=1) 
        x = self.decoder4_iterative_1(torch.cat([x, x_fusion2_iterative_1], dim=1)) 
        x_1 = self.trans_5_iterative_1(x)
        x_2 = self.trans_5_2_iterative_1(x)
        x = self.trans_conv_3_iterative_1(torch.cat([x_1, x_2], dim=1))

        # ----【微分同胚重构】计算二次迭代残差并进行空间复合-----
        delta_velocity_3_iter = self.output_block_3_iterative_1(x)
        delta_flow_3_iter = self.diff[1](delta_velocity_3_iter)
        flow_3 = self.warp[1](flow_3, delta_flow_3_iter) + delta_flow_3_iter 

        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2 


        ## ------------------------------Scale1 (全分辨率: 160x192x224)-------------------------
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  

        # 第五个最终分辨率 decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) 

        # ----【微分同胚重构】最终金字塔顶层速度场集成与复合-----
        x = self.output_block(x) 
        delta_velocity_final = self.flow(x) 
        delta_flow_final = self.diff[0](delta_velocity_final) # 全分辨率积分生成流场

        flow_final = self.warp[0](flow_3_up, delta_flow_final) + delta_flow_final
            
        return flow_1, flow_2, flow_3, flow_final



