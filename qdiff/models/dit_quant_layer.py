import torch
from qdiff.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough
from qdiff.models.quant_layer import QuantLayer

'''
Utility QuantLayers for STDiT temporal/spatial attn layer linears
'''

class QuantAttnLinearImg(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):

        if not self.disable_act_quant and self.act_quant:
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

class QuantCrossAttnLinearImg(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: new forward, cleaner
    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        C = input.shape[2]

        if input.shape[0] == 1:
            layer_type = "kv"
            n_prompt = input.shape[1]
            BS = input.shape[1]//n_prompt
        else:
            layer_type = "q"
            BS = input.shape[0]

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


