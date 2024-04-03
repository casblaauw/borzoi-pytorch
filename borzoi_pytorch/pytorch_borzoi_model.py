# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import numpy as np
import torch
from torch import nn



from .pytorch_borzoi_transformer import Attention

#torch.backends.cudnn.deterministic = True

#torch.set_float32_matmul_precision('high')
  
# --------- COMPONENT MODULES - local ---------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: str = "gelu", conv_type: str = "standard"):
        """"A convolution block, including activation and normalisation.
        
        Arguments:
        - in_channels: integer, number of input features. Should be number of preceding output channels/filters.
        - out_channels: integer, number of filters/kernels/out_channels.
        - kernel_size: integer, width of each filter.
        - activation: str, activation function to use. Only used with conv_type "standard".  One of "gelu",  "relu", "linear", "softplus".
        - conv_type: str, either "standard" or "separable".
        """
        super().__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, groups = in_channels, padding = 'same', bias = False)
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size = 1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(in_channels, eps = 0.001)
            self.activation = get_activator(activation)
            self.conv_layer = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size = kernel_size,
                padding = 'same')    
            
    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x

class ConvBlockPool(ConvBlock):
    """"A convolution+pooling block, including activation and normalisation pre-conv and max-pooling post-conv.
        
        Arguments:
        - in_channels: integer, number of input features. Should be number of preceding output channels/filters.
        - out_channels: integer, number of filters/kernels/out_channels.
        - kernel_size: integer, width of each filter.
        - pool_size: integer, width of the max pooling.
        - activation: str, activation function to use. One of "gelu",  "relu", "linear", "softplus".
        - conv_type: str, either "standard" or "separable".
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int, activation: str = "gelu", conv_type: str = "standard"):
        super().__init__(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, activation = activation, conv_type = conv_type)
        self.pool = nn.MaxPool1d(kernel_size = pool_size, padding = 0)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        x = self.pool(x)
        return x


# --------- COMPONENT MODULES - distal ---------
    
class Transformer(nn.Module):
    def __init__(self, dim: int, key_size: int, heads: int, num_position_features: int, dropout: float, attention_dropout: float = 0.05, position_dropout : float = 0.01, **kwargs):
        super().__init__()
        assert dim % heads == 0
        value_size = dim // heads

        self.norm = nn.LayerNorm(dim, eps = 0.001),
        self.attention = Attention(
                    dim = dim,
                    heads = heads,
                    dim_key = key_size,
                    dim_value = value_size,
                    dropout = attention_dropout,
                    pos_dropout = position_dropout,
                    num_rel_pos_features = num_position_features
                ),
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.attention(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: int, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps = 0.001),
        self.l1 = nn.Linear(dim, dim * 2),
        self.dropout = nn.Dropout(dropout),
        self.activation = nn.ReLU(),
        self.l2 = nn.Linear(dim * 2, dim),

    def forward(self, x):
        x = self.ln(x)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x
    
class TransformerFFN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transf = Transformer(**kwargs)
        self.ff = FeedForward(**kwargs)
    
    def forward(self, x):
        x = self.transf(x)
        x = self.ff(x)
        return x
    
# --------- COMPONENT MODULES - upsampling ---------
class Upscale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: "gelu"):
        """"xxx
        
        Arguments:
        - xxx
        """
        super().__init__()
        self.conv_input = ConvBlock(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, activation = activation)
        self.conv_horizontal = ConvBlock(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, activation = activation)

    def forward(self, x, y):
        x = self.conv_input(x)
        y = self.conv_horizontal(y)


# --------- COMPONENT MODULES - helper ---------
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if target_len == -1:
            return x
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')
        trim = (target_len - seq_len) // 2
        if trim == 0:
            return x
        return x[:, -trim:trim]


# --------- CONSTRUCTION FUNCTIONS ---------   
    
def conv_tower(filters_init: int, filters_end: int, divisible_by: int, kernel_size: int, pool_size: int, repeat: int):
    filter_sizes = exponential_linspace_int(start = filters_init, end = filters_end, num = repeat+1, divisible_by = divisible_by)
    tower_list = [
            ConvBlockPool(in_channels = in_channels, filters = out_channels, kernel_size = kernel_size, pool_size = pool_size)
            for i, (in_channels, out_channels) in enumerate(zip(filter_sizes[:-1, 1:]))
        ]
    return nn.Sequential(*tower_list)
    

def transformer_tower(dim: int, key_size: int, heads: int, num_position_features: int, dropout: float, repeat: int):
    transformer_list = [
        nn.Sequential(
            Residual(Transformer(dim = dim, key_size = key_size, heads = heads, num_position_features = num_position_features, dropout = dropout)),
            Residual(FeedForward(dim = dim, dropout = dropout)))
            for i in range(repeat)
    ]
    return nn.Sequential(*transformer_list)
            

def get_activator(type: str):
    if type.lower() == "gelu":
        return nn.GELU(approximate = 'tanh')
    elif type.lower() == "relu":
        return nn.RELU()
    elif type.lower() == "linear":
        return nn.Identity()
    elif type.lower() == "softplus":
        return nn.Softplus()
    else:
        raise ValueError(f"Did not recognise activation function type {type}.")
    
def get_pooling(type: str, pool_size: int):
    if type.lower() == "maxpool1d":
        return nn.MaxPool1D(kernel_size = pool_size, padding = 0)
    elif type.lower() == "avgpool1d":
        return nn.AvgPool1d(kernel_size = pool_size, padding = 0)
    else:
        raise ValueError(f"Did not recognise pooling function type {type}.")


function_mapping = {
    "Conv1D": nn.Conv1D,
    "ConvBlock": ConvBlock,
    "ConvBlockPool": ConvBlockPool,
    "Transformer": Transformer,
    "FeedForward": FeedForward,
    "TargetLengthCrop": TargetLengthCrop,
    "Dropout": nn.Dropout,
    "conv_tower": conv_tower,
    "transformer_tower": transformer_tower,
    "activation": get_activator,
    "pooling": get_pooling
}


# --------- BORZOI OBJECT ---------    

class Borzoi(nn.Module):
    
    def __init__(self, params):
        #TODO support RC and augs, add gradient functions, and much more
        #TODO rename layers to be understandable if I am feeling like adapting the state dict at some point

        # TODO CAS: add variable num of heads (check modulelist)
        # TODO CAS: add named variables
        # TODO CAS: add docs to all construction functions
        # TODO CAS: add funcs to function_mapping
        # TODO CAS: pick between get_activator and function_mapping approach in mapping
        # TODO CAS: check transposing for attention layers
        # TODO CAS: deal with global params
        # TODO CAS: deal with horizontal layers (hooks somehow?)
        # TODO CAS: add separate load state dict


        super(Borzoi, self).__init__()
        # self.conv_dna = ConvDna(filters = params['trunk']['filters'], kernel_size = 15, pool_size = 2)
        # self.conv_tower = conv_tower()
        self.local = nn.Sequential(*[build_block(block_params) for block_params in params['trunk']['local']])
        self.distal = nn.Sequential(*[build_block(block_params) for block_params in params['trunk']['distal']])
        self.final = nn.Sequential(*[build_block(block_params) for block_params in params['trunk']['final']])
        self.heads = nn.ModuleList(build_block(head_params) for head_params in params['heads'])

        self.trunk = nn.Sequential(self.local, self.distal, self.final)

        # self._max_pool = nn.MaxPool1d(kernel_size = 2, padding = 0)
        # filter_list = exponential_linspace_int(start=filters_init, end=filters_end, num=repeat, divisible_by=divisible_by)
        # self.res_tower = nn.Sequential(
        #     ConvBlock(in_channels = 512, filters = 608, kernel_size = 5),
        #     self._max_pool,
        #     ConvBlock(in_channels = 608, filters = 736, kernel_size = 5),
        #     self._max_pool,
        #     ConvBlock(in_channels = 736, filters = 896, kernel_size = 5),
        #     self._max_pool,
        #     ConvBlock(in_channels = 896, filters = 1056, kernel_size = 5),
        #     self._max_pool,
        #     ConvBlock(in_channels = 1056, filters = 1280, kernel_size = 5)
        # )
        self.unet1 = nn.Sequential(
            self._max_pool,
            ConvBlock(in_channels = 1280, out_channels = 1536, kernel_size = 5)
        )
        # transformer = []
        # for _ in range(8):
        #     transformer.append(nn.Sequential(
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(1536, eps = 0.001),
        #             Attention(
        #                 1536,
        #                 heads = 8,
        #                 dim_key = 64,
        #                 dim_value = 192,
        #                 dropout = 0.05,
        #                 pos_dropout = 0.01,
        #                 num_rel_pos_features = 32
        #             ),
        #             nn.Dropout(0.2))
        #         ),
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(1536, eps = 0.001),
        #             nn.Linear(1536, 1536 * 2),
        #             nn.Dropout(0.2),
        #             nn.ReLU(),
        #             nn.Linear(1536 * 2, 1536),
        #             nn.Dropout(0.2)
        #         )))
        #     )
        self.horizontal_conv0 = ConvBlock(in_channels = 1280, out_channels = 1536, kernel_size = 1)
        self.horizontal_conv1 = ConvBlock(in_channels = 1536, out_channels = 1536, kernel_size = 1)
        self.upsample = torch.nn.Upsample(scale_factor = 2)
        # self.transformer = nn.Sequential(*transformer)
        self.upsampling_unet1 = nn.Sequential(
            ConvBlock(in_channels = 1536, out_channels = 1536, kernel_size = 1),
            self.upsample,
        )
        self.separable1 = ConvBlock(in_channels = 1536, out_channels = 1536,  kernel_size = 3, conv_type = 'separable')
        self.upsampling_unet0 = nn.Sequential(
            ConvBlock(in_channels = 1536, out_channels = 1536, kernel_size = 1),
            self.upsample,
        )
        self.separable0 = ConvBlock(in_channels = 1536, out_channels = 1536,  kernel_size = 3, conv_type = 'separable')
        # self.crop = TargetLengthCrop(16384-32)
        # self.final_joined_convs = nn.Sequential(
        #     ConvBlock(in_channels = 1536, out_channels = 1920, kernel_size = 1),
        #     nn.Dropout(0.1),
        #     nn.GELU(approximate='tanh'),
        # )
        # self.human_head = nn.Conv1d(in_channels = 1920, out_channels = 7611, kernel_size = 1)
        # if self.enable_mouse_head:
        #     self.mouse_head = nn.Conv1d(in_channels = 1920, out_channels = 2608, kernel_size = 1)
        # self.final_softplus = nn.Softplus()

        # self.load_state_dict(torch.load(checkpoint_path))
        
    def forward(self, x):
        # x = self.conv_dna(x)
        # x_unet0 = self.res_tower(x)
        # x_unet1 = self.unet1(x_unet0)
        # x = self._max_pool(x_unet1)
        # x_unet1 = self.horizontal_conv1(x_unet1)
        # x_unet0 = self.horizontal_conv0(x_unet0)
        # x = self.transformer(x.permute(0,2,1))
        # x = x.permute(0,2,1)
        # x = self.upsampling_unet1(x)
        # x += x_unet1
        # x = self.separable1(x)
        # x = self.upsampling_unet0(x)
        # x += x_unet0
        # x = self.separable0(x)
        # x = self.crop(x.permute(0,2,1))
        # x = self.final_joined_convs(x.permute(0,2,1))
        
        # Trunk
        x, skip = self.local(x)
        x = self.distal(x)
        x = self.final(x, skip)

        # Head
        return (head(x) for head in self.heads)
        

        
# --------- MISC FUNCTIONS ---------  

def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]