import torch


class QuantParams:
    def __init__(
        self,
        seq_len: int,
        has_sum_input: bool = False,
        device: torch.device = torch.device("cuda"),
        dtype=torch.float16,
    ):
        self.has_sum_input = has_sum_input

        self.scale_input = torch.empty(
            seq_len,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

        if has_sum_input:
            self.sum_input = torch.empty(
                seq_len,
                dtype=dtype,
                device=device,
                requires_grad=False,
            )
        else:
            self.sum_input = None


# test only, the weight quant process
def quant_weight(state_dict, prefix):
    fp_weight = state_dict[prefix + ".weight"].to(torch.float16)
    weight_max = fp_weight.max(dim=-1).values
    weight_min = fp_weight.min(dim=-1).values
    weight_scale = (weight_max - weight_min) / 255.0
    weight_zp = torch.round((weight_min) / weight_scale) + 128
    weight_quant = torch.clamp(
        torch.round(fp_weight / weight_scale.view(-1, 1)) - weight_zp.view(-1, 1),
        -128,
        127,
    )

    state_dict[prefix + ".weight"] = weight_quant.to(torch.int8)
    state_dict[prefix + ".scale_weight"] = weight_scale.to(torch.float16)
    state_dict[prefix + ".zp_weight"] = weight_zp.to(torch.int16)
