import logging
from types import MethodType
import torch as th
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat

from qdiff.models.quant_layer import QuantLayer
from qdiff.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
# from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
# from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
# from ldm.modules.attention import BasicTransformerBlock
# from ldm.modules.attention import exists, default

# from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity

from functools import partial
from typing import Callable ,Optional, Tuple, Union, Any, Dict
from importlib import import_module

import torch
import torch.nn.functional as F
from torch import einsum, nn

# for diffusers
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import get_activation, GEGLU, GELU, ApproximateGELU
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.resnet import ResnetBlock2D, Upsample2D, Downsample2D
from diffusers.models.attention import BasicTransformerBlock as TransformerBlock
from diffusers.models.attention_processor import *
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.normalization import AdaGroupNorm, AdaLayerNorm, AdaLayerNormZero


# logger = logging.getLogger(__name__)

# ---------------------- QuantResBlock for diffusers ----------------------------------------
class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.weight_quant = False
        self.act_quant = False
        # initialize quantizer
        # self.act_quantizer = ActQuantizer(act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(weight_quant, act_quant)

    def get_quant_state(self):
        return self.weight_quant, self.act_quant

class QuantResnetBlock2D(BaseQuantBlock):
    def __init__(self, res2d: ResnetBlock2D, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.pre_norm = res2d.pre_norm
        self.pre_norm = True
        self.in_channels = res2d.in_channels
        out_channels = self.in_channels if res2d.out_channels is None else res2d.out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = res2d.use_conv_shortcut
        self.up = res2d.up
        self.down = res2d.down
        self.output_scale_factor = res2d.output_scale_factor
        self.time_embedding_norm = res2d.time_embedding_norm
        self.skip_time_act = res2d.skip_time_act

        self.norm1 = res2d.norm1
        self.conv1 = res2d.conv1

        self.time_emb_proj = res2d.time_emb_proj
        self.norm2 = res2d.norm2


        self.dropout = res2d.dropout
        # conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = res2d.conv2

        self.nonlinearity = res2d.nonlinearity

        # self.upsample = self.downsample = None
        self.upsample = res2d.upsample
        self.downsample = res2d.downsample

        self.use_in_shortcut = res2d.use_in_shortcut
        self.conv_shortcut = res2d.conv_shortcut

        # del self.act_quantizer  # created in super class init, donot need

    def forward(
        self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor,  scale: float = 1.0, split: int = 0
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(input_tensor)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(hidden_states)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(input_tensor)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(hidden_states)
            )

        hidden_states = self.conv1(hidden_states) if not USE_PEFT_BACKEND else self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb)[:, :, None, None]
                if not USE_PEFT_BACKEND
                else self.time_emb_proj(temb)[:, :, None, None]
            )

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states) if not USE_PEFT_BACKEND else self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor, split=split) if not USE_PEFT_BACKEND else self.conv_shortcut(input_tensor)
            )

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


# quant transformer block for diffusers
class QuantTransformerBlock(BaseQuantBlock):
    def __init__(self, tran: TransformerBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)


        self.only_cross_attention = tran.only_cross_attention

        self.use_ada_layer_norm_zero = tran.use_ada_layer_norm_zero
        self.use_ada_layer_norm = tran.use_ada_layer_norm
        self.use_ada_layer_norm_single = tran.use_ada_layer_norm_single
        self.use_layer_norm = tran.use_layer_norm

        self.pos_embed = tran.pos_embed

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = tran.norm1
        self.attn1 = tran.attn1

        # 2. Cross-Attn
        self.norm2 = tran.norm2
        self.attn2 = tran.attn2

        # 3. Feed-forward
        self.norm3 = tran.norm3 if hasattr(tran, 'norm3') else None
        self.ff = tran.ff

        # 4. Fuser
        self.fuser = tran.fuser if hasattr(tran, 'fuser') else None

        # 5. Scale-shift for PixArt-Alpha.
        self.scale_shift_table = tran.scale_shift_table if hasattr(tran, 'scale_shift_table') else None

        # let chunk size default to None
        self._chunk_size = tran._chunk_size
        self._chunk_dim = tran._chunk_dim

        # del self.act_quantizer  # created in super class init, donot need

        # the act quantizer for attention QKV output tensor
        self.attn1.act_quantizer_q = ActQuantizer(act_quant_params)
        self.attn1.act_quantizer_k = ActQuantizer(act_quant_params)
        self.attn1.act_quantizer_v = ActQuantizer(act_quant_params)

        self.attn2.act_quantizer_q = ActQuantizer(act_quant_params)
        self.attn2.act_quantizer_k = ActQuantizer(act_quant_params)
        self.attn2.act_quantizer_v = ActQuantizer(act_quant_params)

        # DIRTY: hard coded softmax quantizer in activation (maybe should follow this)
        act_quant_params_softmax = act_quant_params.softmax
        if act_quant_params_softmax is None:
            # skip init act_quantizer
            pass
        else:
            self.attn1.act_quantizer_softmax = ActQuantizer(act_quant_params_softmax)
            self.attn2.act_quantizer_softmax = ActQuantizer(act_quant_params_softmax)

            # INFO: by default, no use quant, use qnn.set_quant_state to turn on
            self.attn1.use_act_quant = False
            self.attn2.use_act_quant = False

        qprocessor = (
            QuantAttnProcessor()
        )
        # different from which in diffusers
        qprocessor1 =  set_qprocessor(self.attn1, qprocessor)
        qprocessor2 =  set_qprocessor(self.attn2, qprocessor)

        self.attn1.processor = qprocessor1
        self.attn2.processor = qprocessor2

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: Union[torch.FloatTensor, Tuple],
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # print("################################ QuantTransformerBlock ####################################")
        if type(hidden_states) == tuple:
            input_tmp = hidden_states
            hidden_states = input_tmp[0]
            attention_mask = input_tmp[1]
            encoder_hidden_states = input_tmp[2]
            encoder_attention_mask = input_tmp[3]
            timestep  = input_tmp[4]
            cross_attention_kwargs = input_tmp[5]
            class_labels = input_tmp[6]

        batch_size = hidden_states.shape[0]

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            # print(type(self.attn2))
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # setting weight quantization here does not affect actual forward pass
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(weight_quant, act_quant)


# quant attention blocks for diffusers
def set_qprocessor(attn: Union["QuantAttention", "Attention"], qprocessor: "QuantAttnProcessor", _remove_lora: bool = False) -> None:
    r"""
    Set the attention processor to use.

    Args:
        processor (`AttnProcessor`):
            The attention processor to use.
        _remove_lora (`bool`, *optional*, defaults to `False`):
            Set to `True` to remove LoRA layers from the model.
    """
    if not USE_PEFT_BACKEND and hasattr(attn, "processor") and _remove_lora and attn.to_q.lora_layer is not None:
        deprecate(
            "set_processor to offload LoRA",
            "0.26.0",
            "In detail, removing LoRA layers via calling `set_default_attn_processor` is deprecated. Please make sure to call `pipe.unload_lora_weights()` instead.",
        )
        # TODO(Patrick, Sayak) - this can be deprecated once PEFT LoRA integration is complete
        # We need to remove all LoRA layers
        # Don't forget to remove ALL `_remove_lora` from the codebase
        for module in attn.modules():
            if hasattr(module, "set_lora_layer"):
                module.set_lora_layer(None)

    # if current processor is in `self._modules` and if passed `processor` is not, we need to
    # pop `processor` from `self._modules`
    if (
        hasattr(attn, "processor")
        and isinstance(attn.processor, torch.nn.Module)
        and not isinstance(qprocessor, torch.nn.Module)
    ):
        logger.info(f"You are removing possibly trained weights of {attn.processor} with {qprocessor}")
        attn._modules.pop("qprocessor")

    return qprocessor


class QuantAttention(BaseQuantBlock):
    def __init__(self, attn: Attention, act_quant_params: dict = {}, qprocessor: Optional["QuantAttnProcessor"] = None,):
        super().__init__(act_quant_params)
        self.attn = attn
        self.inner_dim = attn.inner_dim
        self.cross_attention_dim = attn.cross_attention_dim
        self.upcast_attention = attn.upcast_attention
        self.upcast_softmax = attn.upcast_softmax
        self.rescale_output_factor = attn.rescale_output_factor
        self.residual_connection = attn.residual_connection
        self.dropout = attn.dropout

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = attn._from_deprecated_attn_block

        self.scale_qk = attn.scale_qk
        self.scale = attn.scale

        self.heads = attn.heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = attn.sliceable_head_dim

        self.added_kv_proj_dim = attn.added_kv_proj_dim
        self.only_cross_attention = attn.only_cross_attention

        self.group_norm = attn.group_norm

        self.spatial_norm = attn.spatial_norm

        self.norm_cross = attn.norm_cross


        self.to_q = attn.to_q

        self.to_k = attn.to_k
        self.to_v = attn.to_v

        self.to_out = attn.to_out

        # del self.act_quantizer  # created in super class init, donot need
        if qprocessor is None:
            qprocessor = (
                QuantAttnProcessor()
            )
        # different from which in diffusers
        self.processor =  set_qprocessor(self, qprocessor)

        self.act_quantizer_q = ActQuantizer(act_quant_params)
        self.act_quantizer_k = ActQuantizer(act_quant_params)
        self.act_quantizer_v = ActQuantizer(act_quant_params)

        act_quant_params_softmax = act_quant_params.softmax
        self.act_quantizer_softmax= ActQuantizer(act_quant_params_softmax)


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.attn, name) 


# quant attention processors for diffusers
class QuantAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: QuantAttention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            is_cross_attn = False
            encoder_hidden_states = hidden_states
        else:
            is_cross_attn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        SPLIT_FIRST_TOKEN = False
        # TODO: make cross_attn act_quant skip the 1st SoP token
        # only when cross_atttn, and to_k to_v need to be quantized
        # noted that when to_k have both w and a not quant, the input is placeholder, not actual prompt embedding

        if is_cross_attn and ((attn.to_k.act_quant or attn.to_k.weight_quant) or (attn.to_v.act_quant or attn.to_v.weight_quant)):
            # assert ((attn.to_v.act_quant or attn.to_v.weight_quant) or (attn.to_k.act_quant or attn.to_k.weight_quant))
            SPLIT_FIRST_TOKEN = True

        SPLIT_FIRST_TOKEN = False
        # INFO: split the 1st BoS token, use FP16 
        # (could be resolved by implicitly saving them during actual inference)
        if not SPLIT_FIRST_TOKEN:
            # print("not use split!!!!")
            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)
        else:
            print("use split!!!!")
            exit(0)
            logger.info("split the first token!")
            # encoder_hidden_statets = [BS, N_token , channels]
            key = attn.to_k(encoder_hidden_states[:,1:,:], *args)
            key_first_token = attn.to_k.fwd_func(encoder_hidden_states[:,0,:].unsqueeze(1), attn.to_k.org_weight, attn.to_k.org_bias, **attn.to_k.fwd_kwargs)
            key = torch.cat([key_first_token,key],dim=1)

            value = attn.to_v(encoder_hidden_states[:,1:,:], *args)
            value_first_token = attn.to_v.fwd_func(encoder_hidden_states[:,0,:].unsqueeze(1), attn.to_v.org_weight, attn.to_v.org_bias, **attn.to_v.fwd_kwargs)
            value = torch.cat([value_first_token,value],dim=1)

        # DEBUG_ONLY: disable act_quant of QK_output  !!!
        # if attn.use_act_quant:
            # pass
            # logger.info("should not use the QK output quant")
            # query = attn.act_quantizer_q(query)
            # key = attn.act_quantizer_k(key)
            # value = attn.act_quantizer_v(value)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # DEUBG: disable softmax quantization for now
        # if attn.use_act_quant:
        #     attention_probs = attn.act_quantizer_softmax(attention_probs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def get_specials(model_type):
    if model_type == "opensora" or model_type == "pixart":
        specials = []
    elif model_type == "diffusers":
        specials = {
            # ResBlock: QuantResBlock,
            # BasicTransformerBlock: QuantBasicTransformerBlock,
            # ResnetBlock: QuantResnetBlock,
            # AttnBlock: QuantAttnBlock,
            ResnetBlock2D: QuantResnetBlock2D,
            Attention: QuantAttention,
            TransformerBlock: QuantTransformerBlock,
        }
        # if quant_act:
            # specials[QKMatMul] = QuantQKMatMul
            # specials[SMVMatMul] = QuantSMVMatMul
        # else:
            # specials[AttentionBlock] = QuantAttentionBlock
    else:
        raise NotImplementedError
    return specials


AttentionProcessor = Union[
    AttnProcessor,
    AttnProcessor2_0,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    XFormersAttnAddedKVProcessor,
    CustomDiffusionAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    # deprecated
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
]
