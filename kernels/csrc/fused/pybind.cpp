#include <torch/extension.h>
#include <cuda_fp16.h>


torch::Tensor quant_sum(torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &sum_output, // [tokens]
              torch::Tensor &scaling);

torch::Tensor quant_sum_bf16(torch::Tensor &input,  // [..., hidden_size]
               torch::Tensor &sum_output, // [tokens]
               torch::Tensor &scaling);

torch::Tensor quant_sum_static(torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &sum_output, // [tokens]
              torch::Tensor &scaling);

torch::Tensor gelu_quant_sum(torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &sum_output, // [tokens]
              torch::Tensor &scaling);

void layernorm_nobias(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon);


void layernorm_nobias_quant_nosum_fuse(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon);

void layernorm_nobias_quant_sum_fuse(torch::Tensor &output,    // [batch_size * tokens, hidden_size]
              torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &sum_output, // [batch_size * tokens]
              torch::Tensor &scaling, // [batch_size * tokens]
              float epsilon);

void layernorm_nobias_t2i_fuse(torch::Tensor &output,    // [batch_size * tokens, hidden_size]
              torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &shift_msa, // [batch_size, hidden_size]
              torch::Tensor &scale_msa, // [batch_size, hidden_size]
              float epsilon);

void layernorm_nobias_t2i_quant_sum_fuse(torch::Tensor &output,    // [batch_size * tokens, hidden_size]
              torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &shift_msa, // [batch_size, hidden_size]
              torch::Tensor &scale_msa, // [batch_size, hidden_size]
              torch::Tensor &sum_output, // [batch_size * tokens]
              torch::Tensor &scaling, // [batch_size * tokens]
              float epsilon);

torch::Tensor gate_residual_fuse(torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &gate_msa, // [batch_size, hidden_size]
              torch::Tensor &residual // [batch_size * tokens, hidden_size]
              );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_sum", &quant_sum,
        "quantization kernel, output sum");
  
  m.def("quant_sum_bf16", &quant_sum_bf16,
      "quantization kernel, output sum");

  m.def("quant_sum_static", &quant_sum_static,
        "quantization kernel, output sum");

  m.def("gelu_quant_sum", &gelu_quant_sum,
        "gelu with quantization kernel, output sum");

  m.def("layernorm_nobias", 
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                           float>(
            &layernorm_nobias),
        "layernorm kernel");
  m.def("layernorm_nobias_quant_nosum_fuse", 
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, float>(
            &layernorm_nobias_quant_nosum_fuse),
        "layernorm with quantization kernel");
  
  m.def("layernorm_nobias_t2i_fuse", 
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, torch::Tensor &, float>(
            &layernorm_nobias_t2i_fuse),
        "layernorm with t2i modulate kernel");

  m.def("layernorm_nobias_quant_sum_fuse", 
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, torch::Tensor &, float>(
            &layernorm_nobias_quant_sum_fuse),
        "layernorm with quantization kernel, output sum");

  m.def("layernorm_nobias_t2i_quant_sum_fuse", 
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, float>(
            &layernorm_nobias_t2i_quant_sum_fuse),
        "layernorm with t2i modulate and quantization kernel, output sum");

  m.def("gate_residual_fuse", &gate_residual_fuse,
        "gate msa with residual kernel");
}
