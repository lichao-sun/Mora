# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward, AdaLayerNorm
from rotary_embedding_torch import RotaryEmbedding
from typing import Callable, Optional
from einops import rearrange, repeat

try:
    from diffusers.models.modeling_utils import ModelMixin
except:
    from diffusers.modeling_utils import ModelMixin # 0.11.1


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def exists(x):
    return x is not None


class CrossAttention(nn.Module):
    r"""
    copy from diffuser 0.11.1
    A cross attention layer.
    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        use_relative_position: bool = False,
    ):
        super().__init__()
        # print('num head', heads)
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        # print(use_relative_position)
        self.use_relative_position = use_relative_position
        if self.use_relative_position:
            self.rotary_emb = RotaryEmbedding(min(32, dim_head))
        #     # print(dim_head)
        #     # print(heads)
        #     # adopt https://github.com/huggingface/transformers/blob/8a817e1ecac6a420b1bdc701fcc33535a3b96ff5/src/transformers/models/bert/modeling_bert.py#L265
        #     self.max_position_embeddings = 32
        #     self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, dim_head)

        #     self.dropout = nn.Dropout(dropout)


    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def reshape_for_scores(self, tensor):
        # split heads and dims
        # tensor should be [b (h w)] f (d nd)
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor
    
    def same_batch_dim_to_heads(self, tensor):
        batch_size, head_size, seq_len, dim = tensor.shape # [b (h w)] nd f d
        tensor = tensor.reshape(batch_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, use_image_num=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # [b (h w)] f (nd * d)

        # print('before reshpape query shape', query.shape)
        dim = query.shape[-1]
        if not self.use_relative_position:
            query = self.reshape_heads_to_batch_dim(query) # [b (h w) nd] f d
        # print('after reshape query shape', query.shape)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            if not self.use_relative_position:
                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        # print('query shape', query.shape)
        # print('key shape', key.shape)
        # print('value shape', value.shape)

        if attention_mask is not None:
            # print('attention_mask', attention_mask.shape)
            # print('attention_scores', attention_scores.shape)
            # exit()
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        # print(attention_probs.shape)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)
        # print(attention_probs.shape)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)
        # print(hidden_states.shape)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # print(hidden_states.shape)
        # exit()
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_first_frame: bool = False,
        use_relative_position: bool = False,
        rotary_emb: bool = None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_first_frame=use_first_frame,
                    use_relative_position=use_relative_position,
                    rotary_emb=rotary_emb,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, use_image_num=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        if self.training:
            video_length = hidden_states.shape[2] - use_image_num
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w").contiguous()
            encoder_hidden_states_length = encoder_hidden_states.shape[1]
            encoder_hidden_states_video = encoder_hidden_states[:, :encoder_hidden_states_length - use_image_num, ...]
            encoder_hidden_states_video = repeat(encoder_hidden_states_video, 'b m n c -> b (m f) n c', f=video_length).contiguous()
            encoder_hidden_states_image = encoder_hidden_states[:, encoder_hidden_states_length - use_image_num:, ...]
            encoder_hidden_states = torch.cat([encoder_hidden_states_video, encoder_hidden_states_image], dim=1)
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b m n c -> (b m) n c').contiguous()
        else:
            video_length = hidden_states.shape[2]
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w").contiguous()
            encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length).contiguous()

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                use_image_num=use_image_num,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length + use_image_num).contiguous()
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_first_frame: bool = False,
        use_relative_position: bool = False,
        rotary_emb: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        # print(only_cross_attention)
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        # print(self.use_ada_layer_norm)
        self.use_first_frame = use_first_frame

        # Spatial-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # # SC-Attn
        # self.attn1 = SparseCausalAttention(
        #     query_dim=dim,
        #     heads=num_attention_heads,
        #     dim_head=attention_head_dim,
        #     dropout=dropout,
        #     bias=attention_bias,
        #     cross_attention_dim=cross_attention_dim if only_cross_attention else None,
        #     upcast_attention=upcast_attention,
        # )
        # self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Text Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # # Temp Frame-Cross-Attn; add tahn scale factor
        # self.attn_fcross = SparseCausalAttention(
        #         query_dim=dim,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         cross_attention_dim=cross_attention_dim if only_cross_attention else None,
        #         upcast_attention=upcast_attention,
        # )
        # self.norm_fcross = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        # nn.init.zeros_(self.attn_fcross.to_out[0].weight.data)

        # Temp
        self.attn_temp = TemporalAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=None,
                upcast_attention=upcast_attention,
                rotary_emb=rotary_emb,
            )
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
       

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, op=None):

        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_fcross._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None, use_image_num=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, use_image_num=use_image_num) + hidden_states

        #  # SparseCausal-Attention
        # norm_hidden_states = (
        #     self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        # )

        # if self.only_cross_attention:
        #     hidden_states = (
        #         self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        #     )
        # else:
        #     hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # # Temporal FrameCross Attention
        # norm_hidden_states = (
        #     self.norm_fcross(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_fcross(hidden_states)
        # )
        # hidden_states = self.attn_fcross(
        #     norm_hidden_states, attention_mask=attention_mask, video_length=video_length, use_image_num=use_image_num) + hidden_states
 
        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        # Temporal Attention
        if self.training:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length + use_image_num).contiguous()
            hidden_states_video = hidden_states[:, :video_length, :]
            hidden_states_image = hidden_states[:, video_length:, :]
            # print(hidden_states_video.shape)
            # print(hidden_states_image.shape)
            # if self.training:
            #     # prepare attention mask; mask images in temporal attention
            #     attention_mask_shape = (video_length + use_image_num) // 8 + 1
            #     video_image_length = video_length + use_image_num
            #     attention_mask = torch.zeros([8 * attention_mask_shape, 8 * attention_mask_shape], 
            #                     dtype=hidden_states.dtype, device=hidden_states.device)[:video_image_length, :video_image_length]
            #     attention_mask[:, video_length:] = -math.inf
            norm_hidden_states_video = (
                self.norm_temp(hidden_states_video, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states_video)
            )
            # print(norm_hidden_states.shape)
            hidden_states_video = self.attn_temp(norm_hidden_states_video) + hidden_states_video
            hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d).contiguous()
        else:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length + use_image_num).contiguous()
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            # print(norm_hidden_states.shape)
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d).contiguous()

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        
        return hidden_states


class SparseCausalAttention(CrossAttention):
    def forward_video(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length).contiguous()
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c").contiguous()

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length).contiguous()
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c").contiguous()

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    
    def forward_image(self, hidden_states, encoder_hidden_states=None, attention_mask=None, use_image_num=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # [b (h w)] f (nd * d)
        # if self.use_relative_position:
        #     print('before attention query shape', query.shape)
        dim = query.shape[-1]
        if not self.use_relative_position:
            query = self.reshape_heads_to_batch_dim(query) # [b (h w) nd] f d
        # if self.use_relative_position:
        #     print('before attention query shape', query.shape)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            if not self.use_relative_position:
                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, use_image_num=None):
        if self.training:
            # print(use_image_num)
            hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=video_length + use_image_num).contiguous()
            hidden_states_video = hidden_states[:, :video_length, ...]
            hidden_states_image = hidden_states[:, video_length:, ...]
            hidden_states_video = rearrange(hidden_states_video, 'b f d c -> (b f) d c').contiguous()
            hidden_states_image = rearrange(hidden_states_image, 'b f d c -> (b f) d c').contiguous()
            hidden_states_video = self.forward_video(hidden_states=hidden_states_video, 
                            encoder_hidden_states=encoder_hidden_states, 
                            attention_mask=attention_mask, 
                            video_length=video_length)
            # print('hidden_states_video', hidden_states_video.shape)
            hidden_states_image = self.forward_image(hidden_states=hidden_states_image, 
                                                    encoder_hidden_states=encoder_hidden_states, 
                                                    attention_mask=attention_mask)
            # print('hidden_states_image', hidden_states_image.shape)
            hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=0)
            return hidden_states
            # exit()
        else:
            return self.forward_video(hidden_states=hidden_states, 
                            encoder_hidden_states=encoder_hidden_states, 
                            attention_mask=attention_mask, 
                            video_length=video_length)

class TemporalAttention(CrossAttention):
    def __init__(self, 
                query_dim: int,
                cross_attention_dim: Optional[int] = None,
                heads: int = 8,
                dim_head: int = 64,
                dropout: float = 0.0,
                bias=False,
                upcast_attention: bool = False,
                upcast_softmax: bool = False,
                added_kv_proj_dim: Optional[int] = None,
                norm_num_groups: Optional[int] = None,
                rotary_emb=None):
        super().__init__(query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention, upcast_softmax, added_kv_proj_dim, norm_num_groups)
        # relative time positional embeddings
        self.time_rel_pos_bias = RelativePositionBias(heads=heads, max_distance=32) # realistically will not be able to generate that many frames of video... yet
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        time_rel_pos_bias = self.time_rel_pos_bias(hidden_states.shape[1], device=hidden_states.device)
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # [b (h w)] f (nd * d)
        dim = query.shape[-1]
        
        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask, time_rel_pos_bias)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


    def _attention(self, query, key, value, attention_mask=None, time_rel_pos_bias=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        # print('query shape', query.shape)
        # print('key shape', key.shape)
        # print('value shape', value.shape)
        # reshape for adding time positional bais
        query = self.scale * rearrange(query, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads
        key = rearrange(key, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads
        value = rearrange(value, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads
        # print('query shape', query.shape)
        # print('key shape', key.shape)
        # print('value shape', value.shape)

        # torch.baddbmm only accepte 3-D tensor
        # https://runebook.dev/zh/docs/pytorch/generated/torch.baddbmm
        # attention_scores = self.scale * torch.matmul(query, key.transpose(-1, -2))
        if exists(self.rotary_emb):
            query = self.rotary_emb.rotate_queries_or_keys(query)
            key = self.rotary_emb.rotate_queries_or_keys(key)

        attention_scores = torch.einsum('... h i d, ... h j d -> ... h i j', query, key)
        # print('attention_scores shape', attention_scores.shape)
        # print('time_rel_pos_bias shape', time_rel_pos_bias.shape)
        # print('attention_mask shape', attention_mask.shape)

        attention_scores = attention_scores + time_rel_pos_bias
        # print(attention_scores.shape)

        # bert from huggin face
        # attention_scores = attention_scores / math.sqrt(self.dim_head)

        # # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        if attention_mask is not None:
            # add attention mask
            attention_scores = attention_scores + attention_mask

        # vdm 
        attention_scores = attention_scores - attention_scores.amax(dim = -1, keepdim = True).detach()

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # print(attention_probs[0][0])

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output 
        # hidden_states = torch.matmul(attention_probs, value)
        hidden_states = torch.einsum('... h i j, ... h j d -> ... h i d', attention_probs, value)
        # print(hidden_states.shape)
        # hidden_states = self.same_batch_dim_to_heads(hidden_states)
        hidden_states = rearrange(hidden_states, 'b h f d -> b f (h d)')
        # print(hidden_states.shape)
        # exit() 
        return hidden_states
    
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j') # num_heads, num_frames, num_frames