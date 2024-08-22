import torch
from qdiff.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
from qdiff.models.quant_layer import QuantLayer, find_interval
from omegaconf import ListConfig

'''
Utility QuantLayers for STDiT temporal/spatial attn layer linears
'''

class QuantSpatialAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # check the n_spatial/temporal_token num in act_quant_config is True
        BS = input.shape[0]//self.act_quant_params['n_temporal_token']
        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]
        assert input.shape[1] == S

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,T*S,C])
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*T,S,C])

        if self.weight_quant:
            if self.smooth_quant:
                # during the weight init stage
                if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                    self.weight_quantizer.timestep_wise = True
                    self.weight_quantizer.n_timestep = len(self.timerange)
                    if not self.weight_quantizer.init_done:
                        self.weight_quantizer.delta_list = None  # reset as none for aautomatic init of delta_list during forward
                        self.weight_quantizer.zero_point_list = None  # reset as none for aautomatic init of delta_list during forward
                self.weight_quantizer.cur_timestep_id = cur_timerange_id
                weight = self.weight_quantizer(self.weight * channel_wise_scale)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

class QuantTemporalAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # check the n_spatial/temporal_token num in act_quant_config is True
        BS = input.shape[0]//self.act_quant_params['n_spatial_token']
        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]
        assert input.shape[1] == T

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,S*T,C])
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*S,T,C])

        if self.weight_quant:
            if self.smooth_quant:
                # during the weight init stage
                if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                    self.weight_quantizer.timestep_wise = True
                    self.weight_quantizer.n_timestep = len(self.timerange)
                    if not self.weight_quantizer.init_done:
                        self.weight_quantizer.delta_list = None  # reset as none for aautomatic init of delta_list during forward
                        self.weight_quantizer.zero_point_list = None  # reset as none for aautomatic init of delta_list during forward
                self.weight_quantizer.cur_timestep_id = cur_timerange_id
                weight = self.weight_quantizer(self.weight * channel_wise_scale)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

class QuantCrossAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: new forward, cleaner
    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # Need to handle both Q & KV
        # Q_Linear: [BS, T*S, C]
        # KV_Linear: [1, BS*n_prompt, C]

        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]

        if input.shape[1] == T*S:
            layer_type = "q"
            BS = input.shape[0]
        elif input.shape[0] == 1:
            layer_type = "kv"
            BS = input.shape[1]//self.act_quant_params['n_prompt']
            n_prompt = self.act_quant_params['n_prompt']
        else:
            print('illegeal shape.')
            # import ipdb; ipdb.set_trace()

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)

            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)

            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            if layer_type == 'q':
                input = self.act_quantizer(input)
            elif layer_type == 'kv':
                # INFO: when mask_select=True
                # it only supports dynamic quant
                if not self.act_quant_params.get('dynamic',False):
                    if self.act_quant_params.per_group is False:  # no need to reshape for tensor-wise quant
                        input = self.act_quantizer(input)
                    else:
                        input = input.reshape([BS,n_prompt,C])
                        input = self.act_quantizer(input)
                        input = input.reshape([1,BS*n_prompt,C])
                else:
                    # directly assign N_batch*prompt quant_params for each token
                    input = self.act_quantizer(input)

        if self.weight_quant:
            if self.smooth_quant:
                if not self.weight_quantizer.init_done:
                    if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                        self.weight_quantizer.timestep_wise = True
                        self.weight_quantizer.n_timestep = len(self.timerange)
                        if not self.weight_quantizer.init_done:
                            self.weight_quantizer.delta_list = None  # reset as none for aautomatic init of delta_list during forward
                            self.weight_quantizer.zero_point_list = None  # reset as none for aautomatic init of delta_list during forward
                self.weight_quantizer.cur_timestep_id = cur_timerange_id
                weight = self.weight_quantizer(self.weight * channel_wise_scale)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out


