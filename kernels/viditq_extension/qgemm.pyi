from __future__ import annotations

from torch import Tensor
from typing import Tuple

'''
a stub add for kernels/csrc/qgemm by oneflyingfish
'''

def w8a8_of16_nobias_weight_sym_qserve(
    input: Tensor, weight: Tensor, scale_input: Tensor, scale_weight: Tensor
) -> Tensor: ...
def w8a8_o32(input: Tensor, weight: Tensor) -> Tensor: ...
def w8a8_of16_bias_weight_sym(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    scale_input: Tensor,
    scale_weight: Tensor,
) -> Tensor: ...
def w8a8_bf16_bias_weight_sym(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    scale_input: Tensor,
    scale_weight: Tensor,
) -> Tensor: ...
def w8a8_of16_bias_weight_asym(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    scale_input: Tensor,
    scale_weight: Tensor,
    input_sum: Tensor,
    zp_weight: Tensor,
) -> Tensor: ...
def w8a8_bf16_bias_weight_asym(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    scale_input: Tensor,
    scale_weight: Tensor,
    sum_input: Tensor,
    zp_weight: Tensor,
) -> Tensor: ...
def w4a8_of16_nobias_weight_asym_qserve(
    _in_feats: Tensor,
    _kernel: Tensor,
    _wscales: Tensor,
    _ascales: Tensor,
    _w_szs: Tensor,
    _a_ssums: Tensor,
    _out_feats: Tensor,
) -> None: ...
