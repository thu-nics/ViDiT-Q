#include <torch/extension.h>

torch::Tensor w8a8_of16_nobias_weight_sym_qserve(torch::Tensor input, torch::Tensor weight, torch::Tensor scale_input, torch::Tensor scale_weight);

torch::Tensor w8a8o32(torch::Tensor input,
                      torch::Tensor weight);

torch::Tensor w8a8_o32(torch::Tensor input,
                      torch::Tensor weight);

torch::Tensor w8a8_of16_bias_weight_sym(torch::Tensor input,
                      torch::Tensor weight,
                      torch::Tensor bias,
                      torch::Tensor scale_input,
                      torch::Tensor scale_weight);

torch::Tensor w8a8_of16_bias_weight_asym(torch::Tensor input,
                      torch::Tensor weight,
                      torch::Tensor bias,
                      torch::Tensor scale_input,
                      torch::Tensor scale_weight,
                      torch::Tensor input_sum,
                      torch::Tensor zp_weight);

torch::Tensor w8a8_bf16_bias_weight_asym(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    torch::Tensor scale_input,
                    torch::Tensor scale_weight,
                    torch::Tensor sum_input,
                    torch::Tensor zp_weight);

void w4a8_of16_nobias_weight_asym_qserve(torch::Tensor _in_feats,
                        torch::Tensor _kernel,
                        torch::Tensor _wscales,
                        torch::Tensor _ascales,
                        torch::Tensor _w_szs,
                        torch::Tensor _a_ssums,
                        torch::Tensor _out_feats);