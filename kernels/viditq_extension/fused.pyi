from __future__ import annotations

from torch import Tensor
from typing import Tuple

'''
a stub add for kernels/csrc/fused by oneflyingfish
'''

def quant_sum(input: Tensor, sum_output: Tensor, scaling: Tensor) -> Tensor: ...
def quant_sum_bf16(input: Tensor, sum_output: Tensor, scaling: Tensor) -> Tensor: ...
def quant_sum_static(input: Tensor, sum_output: Tensor, scaling: Tensor) -> Tensor: ...
def gelu_quant_sum(input: Tensor, sum_output: Tensor, scaling: Tensor) -> Tensor: ...
def layernorm_nobias(
    out: Tensor, input: Tensor, weight: Tensor, epsilon: float
) -> None: ...
def layernorm_nobias_quant_nosum_fuse(
    out: Tensor, input: Tensor, weight: Tensor, scaling: Tensor, epsilon: float
) -> None: ...
def layernorm_nobias_quant_sum_fuse(
    output: Tensor,
    input: Tensor,
    weight: Tensor,
    sum_output: Tensor,
    scaling: Tensor,
    epsilon: float,
) -> None: ...
def layernorm_nobias_t2i_fuse(
    output: Tensor,
    input: Tensor,
    weight: Tensor,
    shift_msa: Tensor,
    scale_msa: Tensor,
    epsilon: float,
) -> None: ...
def layernorm_nobias_t2i_quant_sum_fuse(
    output: Tensor,
    input: Tensor,
    weight: Tensor,
    shift_msa: Tensor,
    scale_msa: Tensor,
    sum_output: Tensor,
    scaling: Tensor,
    epsilon: float,
) -> Tuple[Tensor,Tensor]: ...
