import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
from networks.segformer import *
import math
import torch
import torch
import torch.nn as nn
from einops import rearrange

#import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
#import pywt
import torch
import torch.nn as nn
#import pywt
import numbers
import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

from monai.networks.layers.factories import Act, Norm
#   Multi-scale depth-wise convolution (MSDC)\
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class Attention_block(nn.Module):
    """
    Attention block using a gated mechanism.

    This module expects inputs of shape [B, N, C], where N represents a flattened spatial 
    dimension (e.g., a 28x28 feature map flattened to 784 tokens) and C is the number of channels.

    Parameters:
      - F_g: number of channels in input g.
      - F_l: number of channels in input x.
      - F_int: number of internal channels used in the gating mechanism.

    The output is reshaped back to [B, N, F_l].
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int

        # Process g: from [B, F_g, H, W] to [B, F_int, H, W]
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Process x: from [B, F_l, H, W] to [B, F_int, H, W]
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Produce a gating map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        """
        g: [B, N, F_g]
        x: [B, N, F_l]
        """
        B, N, _ = g.size()
        # Dynamically infer spatial dimensions assuming N is a perfect square.
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise ValueError("The flattened spatial dimension is not a perfect square.")

        # Reshape inputs from [B, N, C] to [B, C, H, W]
        g_reshaped = g.permute(0, 2, 1).contiguous().view(B, self.F_g, H, W)
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B, self.F_l, H, W)

        # Apply the 1x1 convolutions followed by batch norm.
        g1 = self.W_g(g_reshaped)   # [B, F_int, H, W]
        x1 = self.W_x(x_reshaped)   # [B, F_int, H, W]

        # Sum and pass through a LeakyReLU activation.
        psi = self.relu(g1 + x1)
        # Generate the gating map (values in [0,1]).
        psi = self.psi(psi)  # [B, 1, H, W]

        # Multiply the gating map with x.
        out = x_reshaped * psi  # [B, F_l, H, W]

        # Reshape the result back to [B, N, F_l]
        out = out.view(B, self.F_l, H * W).permute(0, 2, 1)
        return out
class Attention_block_doubled(nn.Module):
    """
    Modified Attention Block that doubles the output channel dimension.
    Output shape: [B, N, 2 * F_l]
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_doubled, self).__init__()
        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)

        # 🔁 Extra projection layer to double the channel dimension after attention
        self.expand_channels = nn.Sequential(
            nn.Conv2d(F_l, 2 * F_l, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * F_l),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, g, x):
        B, N, _ = g.size()
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise ValueError("The flattened spatial dimension is not a perfect square.")

        g_reshaped = g.permute(0, 2, 1).contiguous().view(B, self.F_g, H, W)
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B, self.F_l, H, W)

        g1 = self.W_g(g_reshaped)
        x1 = self.W_x(x_reshaped)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Apply attention
        out = x_reshaped * psi

        # 🔁 Double the channels
        out = self.expand_channels(out)  # [B, 2*F_l, H, W]

        # Flatten back to [B, N, 2*F_l]
        out = out.view(B, 2 * self.F_l, H * W).permute(0, 2, 1)
        return out

class EfficientCrossAttention(nn.Module):
    """
    Efficient Cross Attention that computes attention in a multi-head, fully vectorized way.
    
    Inputs:
      x1: [B, N, D] -- Lower-level representation (value source)
      x2: [B, N, D'] -- Higher-level representation (key & query source)
      
    The final output has shape [B, N, 2 * value_channels], where value_channels is provided.
    
    Constructor Parameters:
      in_channels:  Input channel dimension (for reference)
      key_channels: Dimension for key (and query) projection.
      value_channels: Dimension for value projection.
      height, width: Spatial dimensions (with N = height * width).
      head_count: Number of attention heads.
    """
    def __init__(self, in_channels, key_channels, value_channels, height, width, head_count=1,token_mlp="mix"):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.head_count = head_count
        self.height = height
        self.width = width

        # Final reprojection: from value_channels to 2 * value_channels.
        self.reprojection = nn.Conv2d(value_channels,  value_channels, kernel_size=1)
        self.norm = nn.LayerNorm( value_channels)
        self.Attn= Attention_block_doubled( value_channels,key_channels ,in_channels)

    def forward(self, x1, x2):
        """
        x1: [B, N, D] (lower-level representation, where D == value_channels)
        x2: [B, N, D'] (higher-level representation, where D' should be equal to key_channels)
        
        """
        
        out1 =self.Attn(x1,x2)
        
        # B, N, _ = x1.size()  # N is expected to equal height*width

        # head_key_channels = self.key_channels // self.head_count
        # head_value_channels = self.value_channels // self.head_count

        # # Rearrange x2 (for keys and queries) from [B, N, key_channels] to [B, head_count, head_key_channels, N]
        # keys = x2.transpose(1, 2).reshape(B, self.head_count, head_key_channels, N)
        # queries = x2.transpose(1, 2).reshape(B, self.head_count, head_key_channels, N)
        # # Rearrange x1 (for values) from [B, N, value_channels] to [B, head_count, head_value_channels, N]
        # values = x1.transpose(1, 2).reshape(B, self.head_count, head_value_channels, N)

        # # Apply softmax over the token dimension for keys and over the channel dimension for queries.
        # keys = F.softmax(keys, dim=-1)
        # queries = F.softmax(queries, dim=2)

        # # Compute attention in a fully vectorized way:
        # # 1. Compute context: [B, head_count, head_key_channels, head_value_channels]
        # context = torch.matmul(keys, values.transpose(-2, -1))
        # # 2. Compute attended values: [B, head_count, head_value_channels, N]
        # attended = torch.matmul(context.transpose(-2, -1), queries)

        # # Merge heads: reshape to [B, value_channels, N]
        # attended = attended.reshape(B, self.value_channels, N)
        # # Reshape into spatial feature map [B, value_channels, height, width]
        # attended = attended.reshape(B, self.value_channels, self.height, self.width)

        # # Apply final reprojection and normalize.
        # reprojected = self.reprojection(attended)  # [B, 2*value_channels, height, width]
        # # Flatten spatial dimensions back to tokens: [B, 2*value_channels, N] then permute to [B, N, 2*value_channels]
        # reprojected = reprojected.reshape(B, self.value_channels, N).permute(0, 2, 1)
        # out = self.norm(reprojected)
       
        
        return  out1#torch.cat([out1, out], dim=-1) # [B, N, 2*value_channels]


class EfficientAttention(nn.Module):
    """
    Optimized Efficient Attention with Flash Attention and Linformer-inspired sequence reduction.
    
    Input  -> x: [B, C, H, W]
    Output ->      [B, C, H, W]
    
    Parameters:
      in_channels:   Input channel dimension (C)
      key_channels:  Dimension of the key (and query) space
      value_channels: Dimension of the value space
      head_count:    Number of attention heads
      downsample:    Whether to apply Linformer-style spatial downsampling
      reduction:     Reduction factor for an extra 1x1 pre-projection (to reduce cost)
    """
    def __init__(self, in_channels, key_channels, value_channels, head_count=1, downsample=True, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.head_count = head_count
        self.downsample = downsample

        # Pre-projection: reduce channels before computing Q, K, V
        reduced_channels = in_channels // reduction
        self.pre_proj = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)

        # Projections for queries, keys, values using the reduced channels
        self.keys = nn.Conv2d(reduced_channels, key_channels, kernel_size=1)
        self.queries = nn.Conv2d(reduced_channels, key_channels, kernel_size=1)
        self.values = nn.Conv2d(reduced_channels, value_channels, kernel_size=1)

        # Linformer-style sequence reduction (if enabled)
        if downsample:
            self.key_downsample = nn.Conv2d(
                key_channels, key_channels, 
                kernel_size=3, stride=2, padding=1, 
                groups=key_channels  # Depthwise convolution
            )
            self.value_downsample = nn.Conv2d(
                value_channels, value_channels,
                kernel_size=3, stride=2, padding=1,
                groups=value_channels
            )

        # Final 1x1 projection: maps the attended features back to the original in_channels
        self.reprojection = nn.Conv2d(value_channels, in_channels, kernel_size=1)

    def forward(self, x):
        
        B, C, H, W = x.shape

        # 0. Pre-projection to reduce input channels (making subsequent operations less expensive)
        x_reduced = self.pre_proj(x)

        # 1. Generate base Q/K/V from the reduced features
        queries = self.queries(x_reduced)
        keys = self.keys(x_reduced)
        values = self.values(x_reduced)

        # 2. Optionally reduce sequence length using Linformer-style downsampling
        if self.downsample:
            keys = self.key_downsample(keys)
            values = self.value_downsample(values)

        # Get spatial dimensions after downsampling (if applied)
        _, _, H_k, W_k = keys.shape

        # 3. Reshape for multi-head attention:
        #     From [B, channels, H, W] to [B, head_count, seq_length, channels_per_head]
        def reshape(tensor, total_channels):
            seq_length = tensor.size(2) * tensor.size(3)
            return tensor.reshape(B, self.head_count, total_channels // self.head_count, seq_length).permute(0, 1, 3, 2)
        
        queries = reshape(queries, self.key_channels)
        keys = reshape(keys, self.key_channels)
        values = reshape(values, self.value_channels)

        # 4. Flash Attention computation
        scale = (self.key_channels // self.head_count) ** -0.5
        attended_values = F.scaled_dot_product_attention(
            queries, keys, values,
            scale=scale,
            dropout_p=0.0,
            is_causal=False
        )

        # 5. Reshape back to spatial format
        attended_values = attended_values.permute(0, 1, 3, 2).reshape(B, self.value_channels, H, W)
        x1 =self.reprojection(attended_values)
        
        # 6. Final projection back to original in_channels
        return x1+x


class EfficientChannelAttention(nn.Module):
    """
    Input  -> x: [B, N, C]
    Output -> [B, N, C]
    
    This module performs channel attention using flash-attention.
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        # Learned scaling parameter per head. Original code multiplies the logits by this.
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #WaveletMLP(in_dim=dim, out_dim=dim*3, wavelet="haar", reduce_factor=2)#
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: [B, N, C] where N is the token (or spatial) dimension and C is the channel dim.
        In channel attention we want to compute attention across the channel dimension.
        """
        x1= x
        B, N, C = x.shape
        head_dim = C // self.num_heads
        
        # Compute Q, K, V. After linear projection and reshape:
        # Shape becomes [B, N, 3, num_heads, head_dim] then permuted to [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is [B, num_heads, N, head_dim]
        
        # For channel attention, we swap the token and channel dimensions so that
        # attention is computed over the channel dimension.
        # After transpose, shapes become [B, num_heads, head_dim, N]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        # Normalize queries and keys along the last dimension (over tokens).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        
        # we scale q accordingly.
        q = q * (self.temperature.unsqueeze(0) * math.sqrt(N))
        
        
        # Call flash-attention. Note: dropout_p expects a float; we extract it from our dropout module.
        attn_out = F.scaled_dot_product_attention(q, k, v, 
                                                  dropout_p=self.attn_drop.p, 
                                                  is_causal=False)
        
        attn_out = attn_out.transpose(-2, -1)
        
        # Merge heads: reshape from [B, num_heads, N, head_dim] to [B, N, C]
        attn_out = attn_out.reshape(B, N, C)
        
        # Final projection and dropout.
        x = self.proj(attn_out)
        x = self.proj_drop(x)
        return x+x1

class InceptionEfficientConv(nn.Module):
    """
    Inception-style convolution module using depthwise separable convolutions.
    This module assumes that the input has already been reduced to a lower channel dimension.
    
    It uses two branches:
      - Branch 1: Depthwise convolution with a 1×3 kernel followed by a 1×1 pointwise conv.
      - Branch 2: Depthwise convolution with a 3×1 kernel followed by a 1×1 pointwise conv.
    The outputs are summed.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branch 1: 1x3 depthwise then 1x1 pointwise
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3),
                      padding=(0, 1), groups=in_channels, bias=False),
            #nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        # Branch 2: 3x1 depthwise then 1x1 pointwise
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1),
                      padding=(1, 0), groups=in_channels, bias=False),
            #nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # x: [B, in_channels, H, W]
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1 + out2

class BottleneckInceptionEfficientConv(nn.Module):
    """
    Bottleneck block that first reduces channels using a 1×1 convolution,
    then applies an InceptionEfficientConv on the reduced representation,
    and finally expands the channels back using another 1×1 convolution.
    
    This reduces the computational cost while still providing the expected output shape.
    
    Parameters:
      - in_channels:  Number of input channels.
      - out_channels: Number of output channels.
      - reduction_ratio: Factor by which channels are reduced. (Default is 2)
    """
    def __init__(self, in_channels, out_channels, reduction_ratio=2):
        super().__init__()
        bottleneck_channels = in_channels // reduction_ratio
        
        # Reduce channels
        self.reduce_conv = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        # Efficient spatial processing on the reduced channels
        self.inception_conv = InceptionEfficientConv(bottleneck_channels, bottleneck_channels)
        # Expand channels back to the desired output
        self.expand_conv = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, in_channels, H, W]
        x = self.reduce_conv(x)       # [B, bottleneck_channels, H, W]
        x = self.inception_conv(x)      # [B, bottleneck_channels, H, W]
        x = self.expand_conv(x)         # [B, out_channels, H, W]
        return x
class ChannelDoubler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_out = nn.Conv2d(
            in_channels=input_size,
            out_channels=input_size ,  # Double the channels
            kernel_size=(1, 1),
            padding=(0, 0),  # No padding for 1x1 kernel
            groups=input_size,  # Group convolution
            bias=False
        )

    def forward(self, x):
        return self.conv_out(x)

#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    
#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class EPA2(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size=None, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1, fake_hw=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.fake_hw = fake_hw  # Required to reshape [B, N, C] to [B, C, H, W]

        # Optional projection layer
        self.input_proj = nn.Linear(hidden_size, hidden_size)

        self.cab = CAB(in_channels=hidden_size, out_channels=hidden_size)
        self.sab = SAB(kernel_size=7)

        self.channel_dropout = nn.Dropout(channel_attn_drop)
        self.spatial_dropout = nn.Dropout(spatial_attn_drop)

        # Optional projection layer back
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)

        #self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        print("x shape: ", x.shape)
        B, N, C = x.shape
        if self.fake_hw is None:
            H = W = int(N ** 0.5)
            assert H * W == N, f"Cannot infer H, W from N={N}. Provide fake_hw=(H, W)."
        else:
            H, W = self.fake_hw
            assert H * W == N, f"Provided fake_hw {self.fake_hw} doesn't match N={N}"

        # Optionally project input
        x_proj = self.input_proj(x)

        # Reshape to image-like format
        x_reshaped = x_proj.permute(0, 2, 1).reshape(B, C, H, W)

        # Channel Attention
        channel_attn = self.cab(x_reshaped)
        x_CA = self.channel_dropout(x_reshaped * channel_attn)

        # Spatial Attention
        spatial_attn = self.sab(x_reshaped)
        x_SA = self.spatial_dropout(x_reshaped * spatial_attn)

        # Fusion
        x_fused = torch.cat([x_CA, x_SA], dim=1)  # Shape: [B, 2C, H, W]
        x_fused = F.adaptive_avg_pool2d(x_fused, (H, W))  # Maintain shape

        # Back to [B, N, C]
        x_out = x_fused.view(B, -1, H * W).permute(0, 2, 1)  # [B, N, 2C]
        x_out = self.output_proj(x_out)  # Project back to [B, N, C]
        print("x shape after CAB/SAB: ", x_out.shape)
        return x_out

import math
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(MSCB, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        
class LightCBAM(nn.Module):
    """
    Lightweight CBAM-style replacement for EPA.
    Preserves signature: [B, N, C] -> [B, N, C]
    """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        reduced = max(4, hidden_size // 16)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(hidden_size, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.ca_drop = nn.Dropout(channel_attn_drop)

        # Spatial attention
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sa_drop = nn.Dropout(spatial_attn_drop)

    def forward(self, x):
       

        # Channel Attention
        ca = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        x = self.ca_drop(x * ca)

        # Spatial Attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.sa_conv(torch.cat([avg, max_], dim=1)))
        x = self.sa_drop(x * sa)
       
        return x
class EPA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size,num_heads=4):
        super().__init__()
        mscb_kernel_sizes=[1, 3, 5]
        qkv_bias=False
        channel_attn_drop=0.1
        spatial_attn_drop=0.1
        self.epa = LightCBAM(input_size, hidden_size, proj_size, num_heads,
                                 qkv_bias, channel_attn_drop, spatial_attn_drop)

        self.hidden_size = hidden_size
        self.reshape_required = True  # controls whether we go to [B, C, H, W]

        self.mscb = MSCB(
            in_channels=hidden_size,
            out_channels=hidden_size,
            stride=1,
            kernel_sizes=mscb_kernel_sizes,
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )

    def forward(self, x):
        #print("x shape before EPA: ", x.shape)
        x = self.epa(x)  
        #print("x shape after EPA: ", x.shape)

       
        x_out = self.mscb(x)  # [B, C, H, W]
        #print("x shape after MSCB: ", x_out.shape)

    

       
        return x_out
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class DualTransformerBlock(nn.Module):
    """
    Transformer Block with Convolutional Substitutions for Efficient Feature Refinement.
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix", num_steps=1):
        super().__init__()
        self.num_steps = num_steps  # Reduce recurrence for efficiency

        #  Convolutional Substitutes
        self.norm1 = nn.LayerNorm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=1)  # Replaces Linear
        #self.pool_ratios = [1,2,4]
        self.attn = EPA(input_size=in_dim, hidden_size=key_dim, proj_size=value_dim, num_heads=4 )#EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=head_count)

        self.norm2 = nn.LayerNorm(in_dim)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=1)  # Replaces Linear
        
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = EfficientChannelAttention(in_dim)

        self.norm4 = nn.LayerNorm(in_dim)
        
        #  MLP → Conv1x1 + Conv3x3
        self.conv_mlp1 = nn.Sequential(
            
            BottleneckInceptionEfficientConv(in_channels=in_dim, out_channels=in_dim)
            #nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, padding=1)
        )
        
        self.conv_mlp2 = nn.Sequential(
           
            BottleneckInceptionEfficientConv(in_channels=in_dim, out_channels=in_dim)
            
        )

        # Adaptive Memory Retention (More Efficient)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable scaling factor

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        """
        Efficient Feature Refinement with Convolutional Enhancements.
        """
        for _ in range(self.num_steps):  # Default 1 iteration
            norm = self.norm1(x)
            norm1 = rearrange(norm, "b (h w) d -> b d h w", h=H, w=W)
            norm1 = self.conv1(norm1)  # Convolutional feature transformation
            # print("norm1 shape: ", norm1.shape)
            attn = self.attn(norm1)
            # print("attn shape: ", attn.shape)
            #attn2 = self.attn2(norm,H,W,d_convs=self.d_convs)
            
            
            attn = rearrange(attn, "b d h w -> b (h w) d")
            #attn= attn+attn2
            add1 = x + attn
            norm2 = self.norm2(add1)
            norm2 = rearrange(norm2, "b (h w) d -> b d h w", h=H, w=W)
            mlp1 = self.conv_mlp1(norm2)  # Convolutional MLP
            mlp1 = rearrange(mlp1, "b d h w -> b (h w) d")

            add2 = add1 + mlp1
            norm3 = self.norm3(add2)
            channel_attn = self.channel_attn(norm3)

            add3 = add2 + channel_attn
            norm4 = self.norm4(add3)
            norm4 = rearrange(norm4, "b (h w) d -> b d h w", h=H, w=W)
            mlp2 = self.conv_mlp2(norm4)  # Convolutional MLP
            mlp2 = rearrange(mlp2, "b d h w -> b (h w) d")

            # Adaptive Memory Retention
            x = x + torch.sigmoid(self.alpha * (add3 + mlp2 - x)) * (add3 + mlp2 - x)

        return x

class WithBias_LayerNorm(nn.Module):
    """
    Layer Normalization with learnable bias and weight.
    """
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear =nn.Linear(x1_dim, out_dim)
            self.cross_attn = EfficientCrossAttention(
                dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn =  EfficientCrossAttention(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn( x2,x1_expand))
            cat_linear_x = cat_linear_x +x1_expand
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out

class FourierAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        _, _, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_fft = self.proj(x_fft.real) + 1j*self.proj(x_fft.imag)
        return torch.fft.irfft2(x_fft, s=(H,W), norm='ortho')

class EfficientWindowAttention(nn.Module):
    """Memory-efficient window attention for medical images"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Shared projections for efficiency
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias for medical anatomy patterns
        self.register_buffer("relative_position_bias", self.create_relative_bias())

    def create_relative_bias(self):
        """Medical-optimized relative position bias"""
        grid = torch.arange(self.window_size)
        relative = grid[:, None] - grid[None, :]
        return nn.Parameter(torch.randn(2*self.window_size-1) * 0.02)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H//self.window_size, self.window_size, 
                 W//self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size**2, C)
        
        # Shared QKV projection
        qkv = self.qkv(x).reshape(-1, self.window_size**2, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Medical-optimized attention
        attn = (q @ k.transpose(-2,-1)) * self.scale + self.relative_position_bias
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1,2).reshape(-1, self.window_size, self.window_size, C)
        
        # Window reconstruction
        x = x.view(B, H//self.window_size, W//self.window_size, 
                 self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return self.proj(x)



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class MiT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs
class DualPathDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims, out_dim, key_dim, value_dim, x1_dim = in_out_chan

        self.use_skip = not is_last
        self.out_dim = out_dim

        self.x1_proj = nn.Linear(x1_dim, out_dim)

        if self.use_skip:
            self.x2_proj = nn.Linear(dims, out_dim)
            self.cross_attn = EfficientCrossAttention(
                dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.fuse_proj = nn.Linear(2 * out_dim, out_dim)
        else:
            self.cross_attn = None

        self.self_refine_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.self_refine_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        if not is_last:
            self.upsample = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.upsample = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2=None):
        b, n, _ = x1.shape
        x1 = self.x1_proj(x1)

        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)  # [B, N, C]
            x2_proj = self.x2_proj(x2)

            skip_attn = self.cross_attn(x2, x1)  # Attend x2 (encoder) to x1 (decoder)
            skip_fused = torch.cat([skip_attn, x2_proj], dim=-1)
            gated_fusion = self.gate(skip_fused) * skip_attn + (1 - self.gate(skip_fused)) * x2_proj
        else:
            gated_fusion = x1
            h = w = int(n ** 0.5)

        path1 = self.self_refine_1(gated_fusion, h, w)
        path2 = self.self_refine_2(x1, h, w)
        fused = path1 + path2  # Could also use concat + proj

        if self.last_layer:
            upsampled = self.upsample(fused).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2)
            return self.last_layer(upsampled)
        else:
            return self.upsample(fused)

class DAEFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        self.backbone = MiT(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )
        self.out1 = nn.Conv2d( in_channels=3, out_channels=9, kernel_size=1,stride=1)
        self.norm =   nn.InstanceNorm2d(9)
        self.out2 = nn.Conv2d( in_channels=9, out_channels=9, kernel_size=1,stride=1)
        self.m = Act['swish']()
    def forward(self, x):
       
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x_app = ((self.out1(x)))
        
        output_enc = self.backbone(x)
        #print("output_enc-----",output_enc[0].shape,output_enc[1].shape,output_enc[2].shape)
        # output_enc[0] = self.norm(output_enc[0])
        # output_enc[0] = self.m(output_enc[0])       

        b, c, _, _ = output_enc[2].shape

        # ---------------Decoder-------------------------
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))
        out = (self.out2(tmp_0+x_app))
        #print("temo-----",tmp_0.shape,tmp_1.shape,tmp_2.shape)  

        return  out
