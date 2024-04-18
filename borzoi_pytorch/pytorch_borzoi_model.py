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
import json
import torch
from torch import nn
from .pytorch_borzoi_transformer import Attention
from copy import deepcopy

#torch.backends.cudnn.deterministic = True

#torch.set_float32_matmul_precision('high')
  
# --------- COMPONENT MODULES - local ---------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: str = "gelu", norm: str = "batchnorm", conv_type: str = "standard"):
        """"A convolution block, including activation and normalisation.
        
        Arguments:
        - in_channels: integer, number of input features. 
            Should be number of preceding output channels/filters.
        - out_channels: integer, number of filters/kernels/out_channels.
        - kernel_size: integer, width of each filter.
        - activation: optional str, activation function to use. 
            Only used with conv_type "standard". 
            One of "gelu",  "relu", "linear", "softplus".
            Default: "gelu".
        - norm: optional str, type of normalisation to use, either "batchnorm" or "layernorm". 
            Default: "batchnorm".
        - conv_type: optional str, either "standard" or "separable".
            Default: "standard".
        """
        super().__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, groups = in_channels, padding = 'same', bias = False)
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size = 1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = get_norm(norm, in_channels = in_channels, eps = 0.001)
            self.activation = get_activation(activation)
            self.conv_layer = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size = kernel_size,
                padding = 'same')    
            
    def forward(self, x):
        """Assumes shape (batch, channels, seq_length)."""
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x

class ConvBlockPool(ConvBlock):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int, pool: str = "maxpool1d", **kwargs):
        """"A convolution+pooling block, including activation and normalisation pre-conv and max-pooling post-conv.
        
        Arguments:
        - in_channels: integer, number of input features. 
            Should be number of preceding output channels/filters.
        - out_channels: integer, number of filters/kernels/out_channels.
        - kernel_size: integer, width of each filter.
        - pool_size: integer, width of the max pooling.
        - pool: optional str, pooling function to use. 
            One of "max"/"maxpool1d" or "avg"/"avgpool1d" (case-insensitive). 
            Default: "maxpool1d".
        - **kwargs: passed on to ConvBlock. 
            Optional kwargs for ConvBlock: activation: str = "gelu", norm: str = "batchnorm", conv_type: str = "standard"
        """
        super().__init__(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, **kwargs)
        self.pool = get_pooling(type = pool, pool_size = pool_size)
    
    def forward(self, x):
        """Assumes shape (batch, channels, seq_length)."""
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        x = self.pool(x)
        return x

class Upscale(nn.Module):
    def __init__(self, in_channels: int, horizontal_in_channels: int, intermed_channels: int, out_channels: int, kernel_size: int, activation: str = "gelu", norm: str = "batchnorm"):
        """"An upscaling block, consisting of:
         - a pointwise standard ConvBlock + Upscaling for the main input (x)
         - a pointwise standard ConvBlock for the horizontal input (y)
         - a separable ConvBlock for the summed inputs.
        
        Arguments:
        - Pointwise ConvBlocks arguments:
            - in_channels: integer, number of input features of the main input. 
                Should be number of preceding output channels/filters.
            - horizontal_in_channels: integer, number of input features of the horizontal/skip input. 
                Should be number of preceding output channels/filters.
            - intermed_channels: integer, output feature size of the pointwise ConvBlocks (which are input into separable ConvBlock).
            - activation: optional str, activation function to use for the standard ConvBlocks. 
                One of "gelu",  "relu", "linear", "softplus".
                Default: "gelu".
        - Separable ConvBlock arguments:
            - out_channels: integer, number of filters of the separable ConvBlock (and therefore of the output of this Upscale block).
            - kernel_size: integer, width of the filters in the separable ConvBlock.
        
        """
        super().__init__()
        self.conv_input = ConvBlock(in_channels = in_channels, out_channels = intermed_channels, kernel_size = 1, activation = activation, norm = norm, conv_type = "standard")
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv_horizontal = ConvBlock(in_channels = horizontal_in_channels, out_channels = intermed_channels, kernel_size = 1, activation = activation, norm = norm, conv_type = "standard")
        self.conv_sep = ConvBlock(in_channels = intermed_channels, out_channels = out_channels, kernel_size = kernel_size, conv_type = 'separable')

    def forward(self, inputs):
        """Assumes shape (batch, channels, seq_length)."""
        x, y = inputs
        x = self.conv_input(x)
        x = self.upsample(x)
        y = self.conv_horizontal(y)
        x += y
        x = self.conv_sep(x)
        return x

# --------- COMPONENT MODULES - distal ---------
    
class MHABlock(nn.Module):
    def __init__(self, dim: int, key_size: int, heads: int, num_position_features: int, dropout: float, attention_dropout: float = 0.05, position_dropout: float = 0.01, norm =  "layernorm"):
        """MHA block (norm+MHA+dropout)

        Arguments:
        - dim: int, input/output channel size. Must be divisible by heads.
        - key_size: int, key size for the attention layer. Value size is inferred from dim and n_heads (dim // heads).
        - heads: int, number of heads in multi-head attention layer.
        - num_position_features: int, number of relative positional features in the attention layer. 
        - dropout: float, dropout to use in the separate dropout layer AFTER attention.
        - attention_dropout: optional float, dropout to use for attention WITHIN the attention layer.
            Default: 0.05.
        - position_dropout: optional float, dropout to use for position encoding WITHIN the attention layer.
            Default: 0.01.
        - norm: optional str, type of normalisation to use, either "layernorm" or "batchnorm". 
            Default: "layernorm", as in Borzoi and original Transformer paper.
        """
        super().__init__()
        assert dim % heads == 0
        value_size = dim // heads

        # self.norm = nn.LayerNorm(dim, eps = 0.001),
        self.norm = get_norm(type = norm, in_channels = dim, eps = 0.001)
        self.attention = Attention(
                    dim = dim,
                    heads = heads,
                    dim_key = key_size,
                    dim_value = value_size,
                    dropout = attention_dropout,
                    pos_dropout = position_dropout,
                    num_rel_pos_features = num_position_features
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Assumes shape (batch, seq_length, channels)."""
        x = self.norm(x)
        x = self.attention(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: int, activation: str = "relu", norm: str = "layernorm"):
        """Feedforward block for use with MHA in the Transformer architecture.
        
        Arguments:
        - dim: int, input/output channel size.
        - dropout: float, dropout to use.
        - activation: optional str, one of "gelu"/"relu"/"linear"/"softplus".
            Default is "relu", as in Borzoi and original Transformer paper.
        - norm: optional str, either "layernorm"/"batchnorm".
            Default is "layernorm", as in Borzoi and original Transformer paper.
        """
        super().__init__()
        self.norm = get_norm(norm, in_channels = dim, eps = 0.001)
        self.l1 = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.l2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        """Assumes shape (batch, seq_length, channels)."""
        x = self.norm(x)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, dim: int, key_size: int, heads: int, num_position_features: int, dropout: float, **kwargs): 
        """Transformer (residual MHABlock + residual FFN) block. 
        Arguments are the same as MHABlock, with dim and dropout also used for FeedForward.
        
        Arguments:
        - dim: int, input/output channel size. Must be divisible by heads.
        - key_size: int, key size for the attention layer. Value size is inferred from dim and n_heads (dim // heads).
        - heads: int, number of heads in multi-head attention layer.
        - num_position_features: int, number of relative positional features in the attention layer. 
        - dropout: float, dropout to use in the separate dropout layer AFTER attention and in the feedforward block.
        - **kwargs: passed on to MHABlock. 
            Optional kwargs for MHABlock: 
            - attention_dropout: optional float, dropout to use for attention WITHIN the attention layer.
            - position_dropout: optional float, dropout to use for position encoding WITHIN the attention layer.
        """
        super().__init__()
        self.transf = MHABlock(dim = dim, key_size = key_size, heads = heads, num_position_features = num_position_features, dropout = dropout, **kwargs)
        self.ff = FeedForward(dim = dim, dropout = dropout)
    
    def forward(self, x):
        """Assumes shape (batch, seq_length, channels)."""
        x2 = self.transf(x)
        x += x2
        x3 = self.ff(x)
        x += x3
        return x
    

# --------- COMPONENT MODULES - helper ---------
    
class Residual(nn.Module):
    def __init__(self, fn):
        """Residual module. Calculates fn(input) and returns input+fn(input).
        
        Arguments:
        - fn: a callable, presumably a nn.Module-ish or nn.Sequential.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class TargetLengthCrop(nn.Module):
    def __init__(self, target_length: int):
        """"Cropping module.
        Works in terms of kept bins (so convert from bp to bins).
        Warning: Borzoi paper arch lists trimmed bins, so:
            Crop(5120) = TargetLengthCrop(524288/32-2*5120) = TargetLengthCrop(6144)
        
        Arguments:
        - target_length: int, number of bins in the center to keep. 
        """
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        """Assumes shape (..., L) i.e. (batch, channels, seq_length).
        """
        seq_len, target_len = x.shape[-1], self.target_length
        if target_len == -1:
            return x
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')
        trim = (target_len - seq_len) // 2
        if trim == 0:
            return x
        return x[..., -trim:trim]
        

# --------- CONSTRUCTION FUNCTIONS ---------   
    
def conv_tower(filters_init: int, filters_end: int, divisible_by: int, repeat: int, **kwargs):
    """Helper function to create a convolution tower with exponentially shifting number of filters/channels.
    Passes all extra arguments to ConvBlockPool.

    Arguments:
    - filters_init: int, 
    - filters_end: int, 
    - divisible_by: int,
    - repeat: int, number of layers.
    - **kwargs: passed on to ConvBlockPool. 
        Required kwargs for ConvBlockPool: kernel_size: int, pool_size: int.
        Optional kwargs for ConvBlockPool: activation: str = "gelu", conv_type: str = "standard"
    
    """
    filter_sizes = exponential_linspace_int(start = filters_init, end = filters_end, num = repeat+1, divisible_by = divisible_by)
    tower_list = [
            ConvBlockPool(in_channels = in_channels, filters = out_channels, **kwargs)
            for i, (in_channels, out_channels) in enumerate(zip(filter_sizes[:-1, 1:]))
        ]
    return nn.Sequential(*tower_list)
    

def transformer_tower(repeat: int, **kwargs):
    """Helper function to create a transformer tower.
    Passes all extra arguments to Transformer.
    
    Arguments;
    - repeat: int, number of layers.
    - **kwargs: passed on to Transformer.
        Required kwargs: dim: int, key_size: int, heads: int, num_position_features: int, dropout: float
        Optional kwargs: attention_dropout: float = 0.05, position_dropout: float = 0.01
    """
    transformer_list = [
        Transformer(**kwargs)
            for i in range(repeat)
    ]
    return nn.Sequential(*transformer_list)
            

def get_activation(type: str) -> nn.Module:
    """Get the specified activation function module.
    
    Arguments:
    - type: str, one of "gelu"/"relu"/"linear"/"softplus" (not case-sensitive).
    """
    type = type.lower()
    if type == "gelu":
        return nn.GELU(approximate = 'tanh')
    elif type == "relu":
        return nn.ReLU()
    elif type == "linear":
        return nn.Identity()
    elif type == "softplus":
        return nn.Softplus()
    else:
        raise ValueError(f"Did not recognise activation function type {type}.")
    
def get_pooling(type: str, pool_size: int) -> nn.Module:
    """Get the specified pooling module.
    
    Arguments:
    - type: str, one of "max"/"maxpool1d"/"avg"/"avgpool1d" (not case-sensitive).
    """
    type = type.lower()
    if type == "maxpool1d" or type == "max":
        return nn.MaxPool1d(kernel_size = pool_size, padding = 0)
    elif type == "avgpool1d" or type == "avg":
        return nn.AvgPool1d(kernel_size = pool_size, padding = 0)
    else:
        raise ValueError(f"Did not recognise pooling function type {type}.")
    
def get_norm(type: str, in_channels: int, eps: float = 0.001):
    """Get the specified normalisation module.

    Arguments:
    - type: str, one of "batch"/"batchnorm"/"batchnorm1d"/"batch-norm" or "layer"/"layernorm"/"layer-norm"
    - in_channels: int, the number of input channels.
    - eps: optional float, value added to the denominator for numerical stability.
        Default: 0.001
    """
    type = type.lower()
    if type in ["batch", "batchnorm", "batchnorm1d", "batch-norm"]:
        return nn.BatchNorm1d(in_channels, eps = eps)
    elif type in ["layer", "layernorm", "layer-norm"]:
        return nn.LayerNorm(in_channels, eps = eps)
    else:
        raise ValueError(f"Did not recognise norm function type {type}.")


function_mapping = {
    "Conv1d": nn.Conv1d,
    "ConvBlock": ConvBlock,
    "ConvBlockPool": ConvBlockPool,
    "Upscale": Upscale,
    "MHABlock": MHABlock,
    "FeedForward": FeedForward,
    "Transformer": Transformer,
    "TargetLengthCrop": TargetLengthCrop,
    "Dropout": nn.Dropout,
    "conv_tower": conv_tower,
    "transformer_tower": transformer_tower,
    "activation": get_activation,
    "pooling": get_pooling,
    "norm": get_norm
}

def build_block(block_params: dict) -> nn.Module:
    """Build a block from a config.
    Arguments:
    - block_params: dict containing 'name' (function name in function_mapping) and block function arguments.
    
    Example: 
    block_params = {
        "name": "ConvBlock",
        "in_channels": 1056,
        "out_channels": 1280,
        "kernel_size": 5
    }
    Which gets turned into:
     function_mapping["ConvBlock"](...) -> ConvBlock(in_channels = 1056, out_channels = 1280, kernel_size = 5)
    """
    block_name = block_params.pop('name')
    return function_mapping[block_name](**block_params)


# --------- BORZOI OBJECT ---------    

class Borzoi(nn.Module):
    
    def __init__(self, params):
        """Create a Borzoi model.
        
        Arguments:
        - params: dict, containing keys 'local'/'distal'/'final'/'heads'.
            Each with specifications for that part of the model, which is passed to self.add_unit.
            If a str, assumed to be a json containing these params, read with json.load().
            
        """
        # TODO: port attention to use new pytorch features (F.scaled_dot_product_attention?)

        # Features currently not supported: layer-specific regularisation, separate initializers, setting activation/norm_type/bn_momentum for all layers
        # Model-wide parameters (l2-scale, syncing batchnorm) should be set in your training code.

        super().__init__()

        if isinstance(params, str):
            with open(params) as params_file:
                params = json.load(params_file)
        else:
            params = params.copy()
        
        assert all(part in params for part in ['local', 'distal', 'final', 'heads']), "Params should contain 'local'/'distal'/'final'/'heads'."

        # Build local (conv tower)
        self.add_unit('local', params['local'])

        # Build distal (transformer tower)
        self.add_unit('distal', params['distal'])

        # Build final of trunk (Upsampling/crop/pointwise)
        self.add_unit('final', params['final'])

        # Build heads
        self.heads = nn.ModuleList()
        self.head_names = []
        for head_name, head_params in params['heads'].items():
            self.add_unit(head_name, head_params)
            self.heads.append(getattr(self, head_name))
            self.head_names.append(head_name)
    
    def add_unit(self, name: str, unit_params):
        """Function to build a borzoi trunk subunit (local/distal/final).
        Adds self.{name} to the model object as a Sequential.
        If the subunit has sublists, those are also made individual Sequentials and saved to self.{name}_list, with self.name as a wrapper.
        This allows for both intermediate output access (by indexing into self.{name}_list) and final output (by using self.{name})

        Arguments:
        - name: str, under which name this unit should be added to the model object.
        - unit_params: dict or list, containing the parameters from that unit, from the params json.
            If a dict (single layer), the single layer is made with build_block. 
            If a list of dicts (list of layers), individual blocks are directly passed to build_block and made Sequential.
            If a list of list of dicts (list of list of layers), the sublists are processed into Sequentials, 
                saved as a ModuleList, and grouped into a big Sequential.

        Single-layer example:
        layer_config = {
            "name": "ConvBlock",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": 5
        }
        self.add_unit('test', layer_config) -> 
            self.test = ConvBlock

        Multi-layer example:
        list_config = [
            layer_config,
            layer_config
        ]
        self.add_unit('test', list_config) -> 
            self.test = nn.Sequential(ConvBlock, ConvBlock)

        Sublist example:
        sublist_config = [
            [
                layer_config,
                layer_config
            ],
            [
                layer_config
            ]
        ] 
        self.add_unit('test', sublist_config) -> 
            self.test_list = nn.ModuleList([nn.Sequential(ConvBlock, ConvBlock), nn.Sequential(ConvBlock)]) 
            self.test = nn.Sequential(nn.Sequential(ConvBlock, ConvBlock), nn.Sequential(ConvBlock))
        """
        # Single layer config:
        # Build as single layer
        if isinstance(unit_params, dict):
            self.__setattr__(
                name,
                build_block(unit_params)
            )
        # List of multiple layer configs:
        else:
            # List of configs: 
            # just build into sequential directly
            if isinstance(unit_params[0], dict):
                self.__setattr__(
                    name,
                    nn.Sequential(*[build_block(block_params) for block_params in unit_params])
                )

            # List of sublists with configs:
            # Save as separate sequentials to self.xxx_list (to allow intermediate access for horizontal layers)
            elif isinstance(unit_params[0], list):
                # Build sequential for each sublist
                sublist_sequentials = []
                for block_params_list in unit_params:
                    sublist_sequentials.append(nn.Sequential(*[build_block(block_params) for block_params in block_params_list]))
                
                # Save list of sequentials as ModuleList
                self.__setattr__(
                    f"{name}_list",
                    nn.ModuleList(sublist_sequentials)
                )
                # Unpack and save total unit as Sequential
                self.__setattr__(
                    name,
                    nn.Sequential(*sublist_sequentials)
                )
            else:
                ValueError(f"Did not recognise config as list of blocks or list of sublists of blocks. Tested params block: {unit_params[0]}")
                
        
    def forward(self, x):
        """Forward pass through the model. 
        Always assumes (batch, seq_len, channels), i.e. (batch, seq_length, 4).
        """
        # Pass through local, saving skip/horizontal tensors
        x = x.permute(0,2,1)
        skip1 = self.local_list[0](x)
        skip2 = self.local_list[1](skip1)
        x = self.local_list[2](skip2)

        # Pass through transformer layer
        x = x.permute(0,2,1)
        x = self.distal(x)
        x = x.permute(0,2,1)
        
        # Pass through final, inserting skip/horizontal tensors
        x = self.final_list[0]((x, skip2))
        x = self.final_list[1]((x, skip1))
        x = self.final_list[2](x)

        # Head
        return [head(x) for head in self.heads]
    
    def forward_trunk(self, x):
        # Local
        skip1 = self.local_list[0](x)
        skip2 = self.local_list[1](skip1)
        x = self.local_list[2](skip2)
        # Distal
        x = x.permute(0,2,1)
        x = self.distal(x)
        x = x.permute(0,2,1)
        # Final
        x = self.final_list[0]((x, skip2))
        x = self.final_list[1]((x, skip1))
        x = self.final_list[2](x)
        return x
    
    def load_weights(self, path):
        """Helper function to load weights from file and do some sanity checks."""
        # Load weights from file
        state_dict = torch.load(path)

        # Load weights into model
        # strict = false because otherwise it complains about no values for alternate versions of lists (self.local, self.final) and heads (self.heads)
        #   which it should copy from the base (self.local_list, self.final_list, self.head_{name}).
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict = False)
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys when loading {path}:\n{unexpected_keys}")

        # Check if weights for alternate versions match original
        assert torch.equal(self.local_list[0][0].weight, self.local[0][0].weight), f"Weights in Borzoi.local_list and Borzoi.local should match. Loading weights probably went wrong."
        for i, head_name in enumerate(self.head_names):
            assert torch.equal(self.heads[i][0].weight, getattr(self, head_name)[0].weight), f"Weights in borzoi's {i+1}th head and Borzoi.heads[{i}] should match. Loading weights probably went wrong."

# --------- MISC FUNCTIONS ---------  

def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]