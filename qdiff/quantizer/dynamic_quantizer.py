import torch
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer, round_ste


'''
The Quantizer that dynamically calculate the quant_params online.
No clipping error online
'''


class DynamicActQuantizer(ActQuantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        assert self.init_done is True   # for dynamic act quantizer, no init_quant_params stage
        assert self.running_stat is False
        assert self.bit_idx == 0

        # INFO: for dynaimc calculateing quant_params, no handling of mixed_precision/timestep_wise, calculating online
        self.init_quant_params(x, self.per_group, momentum=self.running_stat)
        
        self.delta = self.delta_list[self.bit_idx, 0]
        self.zero_point = self.zero_point_list[self.bit_idx, 0]

        # INFO: for dynamic quant, for text_embeds act, may have different input shape
        self.delta_list = None
        self.zero_point_list = None

        assert not torch.all(self.delta == -1) # check if not -1

        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        # start quantization
        # print(f"x shape {x.shape} delta shape {self.delta.shape} zero shape {self.zero_point.shape}")
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        # import ipdb; ipdb.set_trace()
        x_quant_ = self.rounding(x)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant


