import torch
import torch.nn as nn
import viditq_extension.qgemm as qgemm
from viditq_extension.nn.base import QuantParams

class W8A8OF16LinearDynamicInputScale(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "bias",
            torch.empty(
                self.out_features,
                dtype=torch.float16,
                requires_grad=False,
            ) if self.has_bias else None,
        )
        
        self.register_buffer(
            "scale_weight",
            torch.empty(
                self.out_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "zp_weight",
            torch.empty(
                self.out_features,
                dtype=torch.int16,
                requires_grad=False,
            ) if not weight_sym else None,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_sym: bool = True,
        init_only: bool = False,
    ):
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            has_bias=linear.bias is not None,
            weight_sym=weight_sym,
        )

        assert linear.weight.dtype == torch.float16

        if init_only:
            return q_linear

        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)

        ### quantize the weight ###
        fp16_weight = linear.weight.data

        if weight_sym:
            weight_scale = fp16_weight.abs().max(dim=-1).values / 127.0
            weight_quant = torch.clamp(torch.round(fp16_weight / weight_scale.view(-1, 1)), -128, 127)

            q_linear.weight.data[:, :] = weight_quant.to(torch.int8).contiguous()
            q_linear.scale_weight.data[:] = weight_scale.to(torch.float16).contiguous()
            
        else:
            weight_max = fp16_weight.max(dim=-1).values
            weight_min = fp16_weight.min(dim=-1).values

            weight_scale = (weight_max - weight_min) / 255.0
            weight_zp = torch.round((weight_min) / weight_scale) + 128
            weight_quant = torch.clamp(torch.round(fp16_weight / weight_scale.view(-1, 1)) - weight_zp.view(-1, 1), -128, 127)

            q_linear.zp_weight.data[:] = weight_zp.to(torch.int16).contiguous()
            q_linear.weight.data[:, :] = weight_quant.to(torch.int8).contiguous()
            q_linear.scale_weight.data[:] = weight_scale.to(torch.float16).contiguous()

        return q_linear
    
    def forward(self,
        input: torch.Tensor,
        quant_params: QuantParams,
    ):
        # TODO: implement the complete forward pass for other case
        bias = self.bias
        if bias is None:
            bias = torch.zeros(
                self.out_features, dtype=torch.float16, device=input.device
            )
        
        shape = input.shape
        hidden_size = shape[-1]
        output = qgemm.w8a8_of16_bias_weight_asym(
            input.view(-1, hidden_size),
            self.weight,
            bias,
            quant_params.scale_input,
            self.scale_weight,
            quant_params.sum_input,
            self.zp_weight,
        )

        return output.view(*shape[:-1], self.out_features)
    
