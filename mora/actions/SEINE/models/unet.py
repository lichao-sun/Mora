# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import math
import json
import torch
import einops
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

try:
    from diffusers.models.modeling_utils import ModelMixin
except:
    from diffusers.modeling_utils import ModelMixin # 0.11.1

try:
    from .unet_blocks import (
        CrossAttnDownBlock3D,
        CrossAttnUpBlock3D,
        DownBlock3D,
        UNetMidBlock3DCrossAttn,
        UpBlock3D,
        get_down_block,
        get_up_block,
    )
    from .resnet import InflatedConv3d
except:
    from unet_blocks import (
        CrossAttnDownBlock3D,
        CrossAttnUpBlock3D,
        DownBlock3D,
        UNetMidBlock3DCrossAttn,
        UpBlock3D,
        get_down_block,
        get_up_block,
    )
    from resnet import InflatedConv3d

from rotary_embedding_torch import RotaryEmbedding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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
        rel_pos = einops.rearrange(k_pos, 'j -> 1 j') - einops.rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return einops.rearrange(values, 'i j h -> h i j') # num_heads, num_frames, num_frames

@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None, # 64
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_first_frame: bool = False,
        use_relative_position: bool = False,
    ):
        super().__init__()

        # print(use_first_frame)

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # print(only_cross_attention)
        # print(type(only_cross_attention))
        # exit()
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)
            # print(only_cross_attention)
            # exit()

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
            # print(attention_head_dim)
            # exit()

        rotary_emb = RotaryEmbedding(32)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
                use_relative_position=use_relative_position,
                rotary_emb=rotary_emb,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                use_first_frame=use_first_frame,
                use_relative_position=use_relative_position,
                rotary_emb=rotary_emb,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
                use_relative_position=use_relative_position,
                rotary_emb=rotary_emb,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

        # relative time positional embeddings
        self.use_relative_position = use_relative_position
        if self.use_relative_position:
            self.time_rel_pos_bias = RelativePositionBias(heads=8, max_distance=32) # realistically will not be able to generate that many frames of video... yet

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor = None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            # print(emb.shape) # torch.Size([3, 1280])
            # print(class_emb.shape) # torch.Size([3, 1280])
            emb = emb + class_emb

        if self.use_relative_position:
            frame_rel_pos_bias = self.time_rel_pos_bias(sample.shape[2], device=sample.device)
        else:
            frame_rel_pos_bias = None

        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    use_image_num=use_image_num,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, use_image_num=use_image_num,
        )

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    use_image_num=use_image_num,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        # print(sample.shape)

        if not return_dict:
            return (sample,)
        sample = UNet3DConditionOutput(sample=sample)
        return sample
    
    def forward_with_cfg(self, 
                        x, 
                        t, 
                        encoder_hidden_states = None,
                        class_labels: Optional[torch.Tensor] = None,
                        cfg_scale=4.0,
                        use_fp16=False):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, encoder_hidden_states, class_labels).sample
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :4], model_out[:, 4:]
        # eps, rest = model_out[:, :3], model_out[:, 3:] # b c f h w
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, use_concat=False):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

            
        # the content of the config file
        #         {
        # "_class_name": "UNet2DConditionModel",
        # "_diffusers_version": "0.2.2",
        # "act_fn": "silu",
        # "attention_head_dim": 8,
        # "block_out_channels": [
        #     320,
        #     640,
        #     1280,
        #     1280
        # ],
        # "center_input_sample": false,
        # "cross_attention_dim": 768,
        # "down_block_types": [
        #     "CrossAttnDownBlock2D",
        #     "CrossAttnDownBlock2D",
        #     "CrossAttnDownBlock2D",
        #     "DownBlock2D"
        # ],
        # "downsample_padding": 1,
        # "flip_sin_to_cos": true,
        # "freq_shift": 0,
        # "in_channels": 4,
        # "layers_per_block": 2,
        # "mid_block_scale_factor": 1,
        # "norm_eps": 1e-05,
        # "norm_num_groups": 32,
        # "out_channels": 4,
        # "sample_size": 64,
        # "up_block_types": [
        #     "UpBlock2D",
        #     "CrossAttnUpBlock2D",
        #     "CrossAttnUpBlock2D",
        #     "CrossAttnUpBlock2D"
        # ]
        # }
        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        # config["use_first_frame"] = True

        config["use_first_frame"] = False
        if use_concat:
            config["in_channels"] = 9
        # config["use_relative_position"] = True

        # # tmp
        # config["class_embed_type"] = "timestep"
        # config["num_class_embeds"] = 100

        from diffusers.utils import WEIGHTS_NAME # diffusion_pytorch_model.bin
        
        # {'_class_name': 'UNet3DConditionModel', 
        #  '_diffusers_version': '0.2.2', 
        #  'act_fn': 'silu', 
        #  'attention_head_dim': 8, 
        #  'block_out_channels': [320, 640, 1280, 1280], 
        #  'center_input_sample': False, 
        #  'cross_attention_dim': 768, 
        #  'down_block_types': 
        #  ['CrossAttnDownBlock3D', 
        #   'CrossAttnDownBlock3D', 
        #   'CrossAttnDownBlock3D', 
        #   'DownBlock3D'], 
        #   'downsample_padding': 1, 
        #   'flip_sin_to_cos': True, 
        #   'freq_shift': 0, 
        #   'in_channels': 4, 
        #   'layers_per_block': 2, 
        #   'mid_block_scale_factor': 1, 
        #   'norm_eps': 1e-05, 
        #   'norm_num_groups': 32, 
        #   'out_channels': 4, 
        #   'sample_size': 64, 
        #   'up_block_types': 
        #   ['UpBlock3D', 
        #    'CrossAttnUpBlock3D', 
        #    'CrossAttnUpBlock3D', 
        #    'CrossAttnUpBlock3D']}

        model = cls.from_config(config)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")

        if use_concat:
            new_state_dict = {}
            conv_in_weight = state_dict["conv_in.weight"]
            new_conv_weight = torch.zeros((conv_in_weight.shape[0], 9, *conv_in_weight.shape[2:]), dtype=conv_in_weight.dtype)
        
            for i, j in zip([0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8]):
                new_conv_weight[:, j] = conv_in_weight[:, i]
            new_state_dict["conv_in.weight"] = new_conv_weight
            new_state_dict["conv_in.bias"] = state_dict["conv_in.bias"]
            for k, v in model.state_dict().items():
                # print(k)
                if '_temp.' in k:
                    new_state_dict.update({k: v})
                if 'attn_fcross' in k: # conpy parms of attn1 to attn_fcross
                    k = k.replace('attn_fcross', 'attn1')
                    state_dict.update({k: state_dict[k]})
                if 'norm_fcross' in k:
                    k = k.replace('norm_fcross', 'norm1')
                    state_dict.update({k: state_dict[k]})
                    
                if 'conv_in' in k:
                    continue
                else:
                    new_state_dict[k] = v
                # # tmp 
                # if 'class_embedding' in k:
                #     state_dict.update({k: v})
            # breakpoint()
            model.load_state_dict(new_state_dict)
        else:
            for k, v in model.state_dict().items():
                # print(k)
                if '_temp' in k:
                    state_dict.update({k: v})
                if 'attn_fcross' in k: # conpy parms of attn1 to attn_fcross
                    k = k.replace('attn_fcross', 'attn1')
                    state_dict.update({k: state_dict[k]})
                if 'norm_fcross' in k:
                    k = k.replace('norm_fcross', 'norm1')
                    state_dict.update({k: state_dict[k]})

            model.load_state_dict(state_dict)

        return model
