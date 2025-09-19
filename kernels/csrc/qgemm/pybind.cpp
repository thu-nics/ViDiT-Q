#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("w8a8_of16_nobias_weight_sym_qserve", &w8a8_of16_nobias_weight_sym_qserve, "qserve w8a8 gemm kernel");
    m.def("w8a8_o32", &w8a8_o32, "our w8a8o32 kernel");
    m.def("w8a8_of16_bias_weight_sym", &w8a8_of16_bias_weight_sym, "weight symmetric w8a8of16 kernel with input and weigth scale along with bias");
    m.def("w8a8_of16_bias_weight_asym", &w8a8_of16_bias_weight_asym, "weight asymmetric w8a8of16 kernel with input and weigth scale along with bias");
    m.def("w8a8_bf16_bias_weight_asym", &w8a8_bf16_bias_weight_asym, "weight asymmetric w8a8bf16 kernel with input and weigth scale along with bias");
    m.def("w4a8_of16_nobias_weight_asym_qserve", &w4a8_of16_nobias_weight_asym_qserve, "qserve w4a8 gemm kernel");
}

