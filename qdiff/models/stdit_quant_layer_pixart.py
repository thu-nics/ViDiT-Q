import torch
from qdiff.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
from qdiff.models.quant_layer import QuantLayer

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

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,T*S,C])
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*T,S,C])

        if self.weight_quant:
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

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,S*T,C])
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*S,T,C])

        if self.weight_quant:
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

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            if layer_type == 'q':
                input = self.act_quantizer(input)
            elif layer_type == 'kv':
                # INFO: when mask_select=True
                # it only supports dynamic quant
                if not self.act_quant_params.get('dynamic',False):
                    input = input.reshape([BS,n_prompt,C])
                    input = self.act_quantizer(input)
                    input = input.reshape([1,BS*n_prompt,C])
                else:
                    # directly assign N_batch*prompt quant_params for each token
                    input = self.act_quantizer(input)

        if self.weight_quant:
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


