import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time # DEBUG_ONLY

from qdiff.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
from qdiff.quantizer.dynamic_quantizer import DynamicActQuantizer

logger = logging.getLogger(__name__)


def find_interval(timerange, timestep_id):
    for index, interval in enumerate(timerange):
        if interval[0] <= timestep_id <= interval[1]:
            return index
    return None  # If timestep_id is not within any interval


class QuantLayer(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff'):
        super(QuantLayer, self).__init__()
        # self._orginal_module = org_module
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.in_features = org_module.in_features
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        # self.org_weight = org_module.weight.data.clone()
        self.org_weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
            # self.org_bias = org_module.bias.data.clone()
            self.org_bias = org_module.bias
        else:
            self.bias = None
            self.org_bias = None
        self.org_module = org_module

        # set use_quant as False, use set_quant_state to set
        self.weight_quant = False
        self.act_quant = False
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant

        # initialize quantizer
        if self.weight_quant_params is not None:
            self.weight_quantizer = WeightQuantizer(self.weight_quant_params)
        if self.act_quant_params is not None:
            if self.act_quant_params.get('dynamic',False):
                self.act_quantizer = DynamicActQuantizer(self.act_quant_params)
            else:
                self.act_quantizer = ActQuantizer(self.act_quant_params)
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr
        # for smooth quant
        smooth_quant_params = act_quant_params.get("smooth_quant", {})
        self.smooth_quant = smooth_quant_params.get("enable", False)
        if self.smooth_quant:
            self.act_quantizer.register_buffer("act_scale", None)
            self.channel_wise_scale_type = smooth_quant_params.get("channel_wise_scale_type", "dynamic")
            self.smooth_quant_momentum = smooth_quant_params.get("momentum", 0)
            self.smooth_quant_alpha = smooth_quant_params.get("alpha", None)
            self.smooth_quant_running_stat = False

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0, smooth_quant_enable: bool = False):
        # DEBUG_ONLY: test the time of init
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()
            
        if self.smooth_quant:
            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(self.smooth_quant_alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - self.smooth_quant_alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = cur_act_scale
                    else:
                        self.act_quantizer.act_scale = self.act_quantizer.act_scale * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale is not None
                channel_wise_scale = self.act_quantizer.act_scale.pow(self.smooth_quant_alpha) / self.weight.abs().max(dim=0)[0].pow(1 - self.smooth_quant_alpha)
            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            # for timeranges, update the act_scale for each timerange respectively
            # if not hasattr(self, 'timerange'):
            #     cur_timerange_id = 0
            # else:
            #     cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([*cur_act_scale.shape]).to(input.dtype)
                if self.act_quantizer.act_scale.abs().mean()==0:
                    self.act_quantizer.act_scale = cur_act_scale
                else:
                    self.act_quantizer.act_scale = self.act_quantizer.act_scale* self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)


        if not self.disable_act_quant and self.act_quant:
            if self.split != 0:
                if self.act_quant_mode == 'qdiff':
                    input_0 = self.act_quantizer(input[:, :self.split, :, :])
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                if self.act_quant_mode == 'qdiff':
                    input = self.act_quantizer(input)

        if self.weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                if self.smooth_quant:
                    weight = self.weight_quantizer(self.weight * channel_wise_scale)
                else:
                    weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            if self.smooth_quant:
                weight = self.org_weight * channel_wise_scale
            else:
                weight = self.org_weight
            bias = self.org_bias
            
        # if self.smooth_quant:
        #     import ipdb; ipdb.set_trace()
        # t_quantizer_init_done = time.time()
        # logging.info('quantizer init elapsed time:{}'.format(t_quantizer_init_done - t_start))

        # if(type(self.fwd_func)==F.linear):
        #     print(input.shape, weight.shape)

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        # DEBUG_ONLY: print the dtype
        # if bias == None:
            # print(input.dtype, weight.dtype)
        # else:
            # print(input.dtype, weight.dtype, bias.dtype)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)  # 在输出的channel上进行channel_wise的量化
        out = self.activation_function(out)
        # logging.info('module forward elapsed time:{}'.format(time.time() - t_quantizer_init_done))
        # import ipdb; ipdb.set_trace()

        # torch.cuda.empty_cache()  # DEBUG: memory accumulate

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):  # 判断是否设置为量化模式！！！
        self.weight_quant = weight_quant
        self.act_quant = act_quant

    def get_quant_state(self):
        return self.weight_quant, self.act_quant

    def set_split(self):
        self.weight_quantizer_0 = WeightQuantizer(self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = ActQuantizer(self.act_quant_params)

    # def set_running_stat(self, running_stat: bool):
        # if self.act_quant_mode == 'qdiff':
            # self.act_quantizer.running_stat = running_stat
            # if self.split != 0:
                # self.act_quantizer_0.running_stat = running_stat

    # def __getattr__(self, name):
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self._orginal_module, name) 

    # def __getattr__(self, name: str) -> Union[torch.Tensor, 'Module']:
    #     return self._orginal_module.__getattr__(name)
