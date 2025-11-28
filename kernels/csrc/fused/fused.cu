#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "../dispatch_utils.h"
#include "../utils.cuh"
#include "../reduction_utils.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

enum class SumType
{
    kNone,
    kPreQuant,
    kPostQuant
};

enum class LoadType
{
  kFloat2,
  kFloat4,
};

template <typename T> __device__ __forceinline__ T gelu_func(const T &x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

// TODO: ActQuantKernel

template<typename DTypeLoad = float2, SumType sum_type = SumType::kNone, bool dynamic = true>
__global__ void QuantKernelBF16(const __nv_bfloat16 *__restrict__ input,
                            int8_t *__restrict__ output,
                            __nv_bfloat16 *__restrict__ sum_output,
                            __nv_bfloat16 *__restrict__ scale,
                            int num_tokens, int hidden_size)
{
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  constexpr int n_packed = (sizeof(DTypeLoad) / sizeof(__nv_bfloat162));
  static_assert(n_packed == 2 || n_packed == 4);
  const int j = (n_packed == 2) ? (tidx << 2) : (tidx << 3);

  __nv_bfloat162 x_val[n_packed];
  *(DTypeLoad*)(&x_val[0]) = *(DTypeLoad*)(&input[bidx * hidden_size + j]);

  if (sum_type == SumType::kPreQuant) {
    float local_sum = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < n_packed; i++) {
      local_sum += __bfloat162float(x_val[i].x);
      local_sum += __bfloat162float(x_val[i].y);
    }

    float sum = vllm::blockReduceSum(local_sum);
    if (tidx == 0) {
      sum_output[bidx] = __float2bfloat16_rn(sum);
    }
  }

  __shared__ float s_amax;

  if constexpr (dynamic) {
    float amax_val = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < n_packed; i++) {
      amax_val = fmaxf(amax_val,
                 fabsf(__bfloat162float(x_val[i].x)));
      amax_val = fmaxf(amax_val,
                 fabsf(__bfloat162float(x_val[i].y)));

    }

    float block_amax_val = vllm::blockReduceMax(amax_val);
    if (tidx == 0) {
      s_amax = block_amax_val;
      scale[bidx] = __float2bfloat16_rn(block_amax_val / 127.0f);
    }
  } else {
    if (tidx == 0) {
      s_amax = __bfloat162float(scale[bidx]);
    }
  }

  __syncthreads();
  float tmp_scale = 127.0f / s_amax;

  char4 o_val[n_packed / 2];

#pragma unroll
  for (uint32_t i = 0; i < n_packed; i += 2) {
    o_val[i / 2] = make_char4(
      float_to_int8_rn(__bfloat162float(x_val[i].x) * tmp_scale),
      float_to_int8_rn(__bfloat162float(x_val[i].y) * tmp_scale),
      float_to_int8_rn(__bfloat162float(x_val[i + 1].x) * tmp_scale),
      float_to_int8_rn(__bfloat162float(x_val[i + 1].y) * tmp_scale)
    );
  }

  if constexpr (sum_type == SumType::kPostQuant) {
    int32_t local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < n_packed / 2; i++) {
      local_sum += static_cast<int32_t>(o_val[i].x);
      local_sum += static_cast<int32_t>(o_val[i].y);
      local_sum += static_cast<int32_t>(o_val[i].z);
      local_sum += static_cast<int32_t>(o_val[i].w);
    }

    int32_t sum = vllm::blockReduceSum(local_sum);
    if (tidx == 0) {
      sum_output[bidx] = __float2bfloat16_rn(__int2float_rn(sum) / tmp_scale);
    }
  }

  using OutputType = typename std::conditional<n_packed == 2, uint32_t, uint64_t>::type;
  *reinterpret_cast<OutputType*>(&output[bidx * hidden_size + j]) = *reinterpret_cast<OutputType*>(&o_val);
}


template<typename DTypeLoad=float2, SumType sum_type=SumType::kNone, bool dynamic=true>
__global__ void QuantKernel(const half *__restrict__ input,
                             int8_t *__restrict__ output, half *__restrict__ sum_output, half *__restrict__ scale,
                             int num_tokens, int hidden_size) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  constexpr int n_packed = (sizeof(DTypeLoad) / sizeof(half2));
  static_assert(n_packed == 2 || n_packed == 4);
  const int j = (n_packed == 2) ? (tidx << 2) : (tidx << 3);

  half2 x_val[n_packed];
  *(DTypeLoad*)(&x_val[0]) = *(DTypeLoad*)(&input[bidx * hidden_size + j]);

  // calculate sum using original value
  if (sum_type == SumType::kPreQuant)
  {
    float local_sum = 0.0f;
    float sum = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < n_packed; i++)
    {
      local_sum += __half2float(x_val[i].x);
      local_sum += __half2float(x_val[i].y);
    }

    sum = vllm::blockReduceSum(local_sum);
    if (tidx == 0)
    {
      sum_output[bidx] = __float2half_rn(sum);
    }
  }

  __shared__ float s_amax;

  if constexpr (dynamic)
  {
    float amax_val = 0.0f;

  #pragma unroll
    for (uint32_t i = 0; i < n_packed; i++)
    {
      amax_val = fmaxf(amax_val, __half2float(__habs(x_val[i].x)));
      amax_val = fmaxf(amax_val, __half2float(__habs(x_val[i].y)));
    }

    const float block_amax_val = vllm::blockReduceMax(amax_val);
    if (tidx == 0) {
      s_amax = block_amax_val;
      scale[bidx] = __float2half_rn(block_amax_val / 127.0f);
    }
  }
  else
  {
    if (tidx == 0) {
      s_amax = scale[bidx];
    }
  }

  __syncthreads();

  float tmp_scale = 127.0f / s_amax;

  char4 o_val[n_packed / 2];

#pragma unroll
  for (uint32_t i = 0; i < n_packed; i += 2)
  {
    o_val[i / 2] = make_char4(
      float_to_int8_rn((float)x_val[i].x * tmp_scale),
      float_to_int8_rn((float)x_val[i].y * tmp_scale),
      float_to_int8_rn((float)x_val[i + 1].x * tmp_scale),
      float_to_int8_rn((float)x_val[i + 1].y * tmp_scale)
    );
  }

  // calculate sum using reconstructed value
  if constexpr (sum_type == SumType::kPostQuant)
  {
    int32_t local_sum = 0;
    int32_t sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < n_packed / 2; i++)
    {
      local_sum += static_cast<int32_t>(o_val[i].x);
      local_sum += static_cast<int32_t>(o_val[i].y);
      local_sum += static_cast<int32_t>(o_val[i].z);
      local_sum += static_cast<int32_t>(o_val[i].w);
    }

    sum = vllm::blockReduceSum(local_sum);
    if (tidx == 0)
    {
      sum_output[bidx] = __float2half_rn(__int2float_rn(sum) / tmp_scale);
    }
  }

  // int8 result
  using OutputType = typename std::conditional<n_packed == 2, uint32_t, uint64_t>::type;
  *reinterpret_cast<OutputType*>(&output[bidx * hidden_size + j]) = *reinterpret_cast<OutputType*>(&o_val);
}


template<typename DTypeLoad=float2, SumType sum_type=SumType::kNone>
__global__ void GeluQuantFuse(const half *__restrict__ input,
                             int8_t *__restrict__ output, half *__restrict__ sum_output, half *__restrict__ scale,
                             int num_tokens, int hidden_size) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  constexpr int n_packed = (sizeof(DTypeLoad) / sizeof(half2));
  static_assert(n_packed == 2 || n_packed == 4);
  const int j = (n_packed == 2) ? (tidx << 2) : (tidx << 3);

  half2 x_val[n_packed];
  *(DTypeLoad*)(&x_val[0]) = *(DTypeLoad*)(&input[bidx * hidden_size + j]);

  // gelu
#pragma unroll
  for (uint32_t i = 0; i < n_packed; i++)
  {
    x_val[i].x = gelu_func(x_val[i].x);
    x_val[i].y = gelu_func(x_val[i].y);
  }

  // calculate sum using original value
  if (sum_type == SumType::kPreQuant)
  {
    float local_sum = 0.0f;
    float sum = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < n_packed; i++)
    {
      local_sum += __half2float(x_val[i].x);
      local_sum += __half2float(x_val[i].y);
    }

    sum = vllm::blockReduceSum(local_sum);
    if (tidx == 0)
    {
      sum_output[bidx] = __float2half_rn(sum);
    }
  }

  float amax_val = 0.0f;

#pragma unroll
  for (uint32_t i = 0; i < n_packed; i++)
  {
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[i].x)));
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[i].y)));
  }

  __shared__ float s_amax;
  const float block_amax_val = vllm::blockReduceMax(amax_val);
  if (tidx == 0) {
    s_amax = block_amax_val;
    scale[bidx] = __float2half_rn(block_amax_val / 127.0f);
  }
  __syncthreads();

  float tmp_scale = 127.0f / s_amax;

  char4 o_val[n_packed / 2];

#pragma unroll
  for (uint32_t i = 0; i < n_packed; i += 2)
  {
    o_val[i / 2] = make_char4(
      float_to_int8_rn((float)x_val[i].x * tmp_scale),
      float_to_int8_rn((float)x_val[i].y * tmp_scale),
      float_to_int8_rn((float)x_val[i + 1].x * tmp_scale),
      float_to_int8_rn((float)x_val[i + 1].y * tmp_scale)
    );
  }

  // calculate sum using reconstructed value
  if constexpr (sum_type == SumType::kPostQuant)
  {
    int32_t local_sum = 0;
    int32_t sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < n_packed / 2; i++)
    {
      local_sum += static_cast<int32_t>(o_val[i].x);
      local_sum += static_cast<int32_t>(o_val[i].y);
      local_sum += static_cast<int32_t>(o_val[i].z);
      local_sum += static_cast<int32_t>(o_val[i].w);
    }

    sum = vllm::blockReduceSum(local_sum);
    if (tidx == 0)
    {
      sum_output[bidx] = __float2half_rn(__int2float_rn(sum) / tmp_scale);
    }
  }

  // int8 result
  using OutputType = typename std::conditional<n_packed == 2, uint32_t, uint64_t>::type;
  *reinterpret_cast<OutputType*>(&output[bidx * hidden_size + j]) = *reinterpret_cast<OutputType*>(&o_val);
}

template <bool has_t2i=false, bool quant=false, SumType sum_type=SumType::kNone>
__global__ void LayernormT2iQuantFuse(const half *__restrict__ input, const half *__restrict__ gamma,  int8_t *__restrict__ normed_output_quant, half *__restrict__ normed_output, half *__restrict__ sum_output, const half *__restrict__ shift_msa, const half *__restrict__ scale_msa, const float eps,
    const int shift_stride, const int scale_stride, const int batch_num_rows, const int hidden_dim, half* __restrict__ scale)
{
  __shared__ float s_mean;
  __shared__ float s_variance;

  half2 x_val[2];
  half2 w_val[2];

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  int j = tidx << 2;

  const int bz = bidx / batch_num_rows;

  float mean = 0.0f;
  float variance = 0.0f;
  float local_sum = 0.0f;
  float local_var_sum = 0.0f;

  *(float2*)(&x_val[0]) = *(float2*)(&input[bidx * hidden_dim + j]);
  *(float2*)(&w_val[0]) = *(float2*)(&gamma[j]);

  local_sum += __half2float(x_val[0].x);
  local_sum += __half2float(x_val[0].y);
  local_sum += __half2float(x_val[1].x);
  local_sum += __half2float(x_val[1].y);

  mean = vllm::blockReduceSum(local_sum);

  // TODO: whether to use reduce or all reduce?
  if (threadIdx.x == 0)
  {
    mean = mean / hidden_dim;
    s_mean = mean;
  }
  __syncthreads();

  mean = s_mean;

  local_var_sum += (__half2float(x_val[0].x) - mean) * (__half2float(x_val[0].x) - mean);
  local_var_sum += (__half2float(x_val[0].y) - mean) * (__half2float(x_val[0].y) - mean);
  local_var_sum += (__half2float(x_val[1].x) - mean) * (__half2float(x_val[1].x) - mean);
  local_var_sum += (__half2float(x_val[1].y) - mean) * (__half2float(x_val[1].y) - mean);

  variance = vllm::blockReduceSum(local_var_sum);

  if (threadIdx.x == 0)
  {
    s_variance = rsqrtf(variance / hidden_dim + eps);
  }
  __syncthreads();

  // abuse this variable. this is actually the reciprocal of the standard deviation
  variance = s_variance;

  x_val[0].x = __float2half_rn((__half2float(x_val[0].x) - mean) * variance * __half2float(w_val[0].x));
  x_val[0].y = __float2half_rn((__half2float(x_val[0].y) - mean) * variance * __half2float(w_val[0].y));
  x_val[1].x = __float2half_rn((__half2float(x_val[1].x) - mean) * variance * __half2float(w_val[1].x));
  x_val[1].y = __float2half_rn((__half2float(x_val[1].y) - mean) * variance * __half2float(w_val[1].y));

  // t2i modulate
  if constexpr (has_t2i) {
    // reuse register w_val
    const __half2 one_half2 = __halves2half2(__float2half(1.0f), __float2half(1.0f));
    *(float2*)(&w_val[0]) = *(float2*)(&scale_msa[bz * scale_stride + j]);
    x_val[0] = __hmul2(x_val[0], __hadd2(w_val[0], one_half2));
    x_val[1] = __hmul2(x_val[1], __hadd2(w_val[1], one_half2));

    *(float2*)(&w_val[0]) = *(float2*)(&shift_msa[bz * shift_stride + j]);
    x_val[0] = __hadd2(x_val[0], w_val[0]);
    x_val[1] = __hadd2(x_val[1], w_val[1]);
  }

  // quantize
  if constexpr (quant) {

    // calculate sum using original value
    if (sum_type == SumType::kPreQuant)
    {
      float local_sum = 0.0f;
      float sum = 0.0f;

      local_sum += __half2float(x_val[0].x);
      local_sum += __half2float(x_val[0].y);
      local_sum += __half2float(x_val[1].x);
      local_sum += __half2float(x_val[1].y);

      sum = vllm::blockReduceSum(local_sum);
      if (tidx == 0)
      {
        sum_output[bidx] = __float2half_rn(sum);
      }
    }

    float amax_val = 0.0f;

    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[0].x)));
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[0].y)));
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[1].x)));
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[1].y)));

    __shared__ float s_amax;
    const float block_amax_val = vllm::blockReduceMax(amax_val);
    if (tidx == 0) {
      s_amax = block_amax_val;
      scale[bidx] = __float2half_rn(block_amax_val / 127.0f);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;

    char4 o_val = make_char4(
      float_to_int8_rn((float)x_val[0].x * tmp_scale),
      float_to_int8_rn((float)x_val[0].y * tmp_scale),
      float_to_int8_rn((float)x_val[1].x * tmp_scale),
      float_to_int8_rn((float)x_val[1].y * tmp_scale)
    );

    // calculate sum using reconstructed value
    if constexpr (sum_type == SumType::kPostQuant)
    {
      int32_t local_sum = 0;
      int32_t sum = 0;

      local_sum += static_cast<int32_t>(o_val.x);
      local_sum += static_cast<int32_t>(o_val.y);
      local_sum += static_cast<int32_t>(o_val.z);
      local_sum += static_cast<int32_t>(o_val.w);

      sum = vllm::blockReduceSum(local_sum);
      if (tidx == 0)
      {
        sum_output[bidx] = __float2half_rn(__int2float_rn(sum) / tmp_scale);
      }
    }

    // int8 result
    *reinterpret_cast<char4*>(&normed_output_quant[bidx * hidden_dim + j]) = o_val;
  }
  else
  {
    // fp16 result
    *(float2*)(&normed_output[bidx * hidden_dim + j]) = *(float2*)(&x_val[0]);
  }
}

template<bool has_residual=false, bool quant=false, SumType sum_type=SumType::kNone>
__global__ void GateResidualQuantFuse(const half *__restrict__ input, const half *__restrict__ gate_msa, const half *__restrict__ residual, int8_t *__restrict__ output_quant, half *__restrict__ output, half *__restrict__ sum_output,
    const int gate_stride, const int batch_num_rows, const int hidden_dim, half *__restrict__ scale, half *__restrict__ intermediate_result)
{
  half2 x_val[2];
  half2 w_val[2];

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  int j = tidx << 2;

  const int bz = bidx / batch_num_rows;

  *(float2*)(&x_val[0]) = *(float2*)(&input[bidx * hidden_dim + j]);
  *(float2*)(&w_val[0]) = *(float2*)(&gate_msa[bz * gate_stride + j]);

  x_val[0] = __hmul2(x_val[0], w_val[0]);
  x_val[1] = __hmul2(x_val[1], w_val[1]);

  *(float2*)(&intermediate_result[bidx * hidden_dim + j]) = *(float2*)(&x_val[0]);

  // residual
  if constexpr (has_residual)
  {
    // abuse w_val
    *(float2*)(&w_val[0]) = *(float2*)(&residual[bidx * hidden_dim + j]);

    x_val[0] = __hadd2(x_val[0], w_val[0]);
    x_val[1] = __hadd2(x_val[1], w_val[1]);
  }

  // quantize
  if constexpr (quant)
  {

    // calculate sum using original value
    if (sum_type == SumType::kPreQuant)
    {
      float local_sum = 0.0f;
      float sum = 0.0f;

      local_sum += __half2float(x_val[0].x);
      local_sum += __half2float(x_val[0].y);
      local_sum += __half2float(x_val[1].x);
      local_sum += __half2float(x_val[1].y);

      sum = vllm::blockReduceSum(local_sum);
      if (tidx == 0)
      {
        sum_output[bidx] = __float2half_rn(sum);
      }
    }

    float amax_val = 0.0f;

    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[0].x)));
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[0].y)));
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[1].x)));
    amax_val = fmaxf(amax_val, __half2float(__habs(x_val[1].y)));

    __shared__ float s_amax;
    const float block_amax_val = vllm::blockReduceMax(amax_val);
    if (tidx == 0) {
        s_amax = block_amax_val;
        scale[bidx] = __float2half_rn(block_amax_val / 127.0f);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;

    char4 o_val = make_char4(
        float_to_int8_rn((float)x_val[0].x * tmp_scale),
        float_to_int8_rn((float)x_val[0].y * tmp_scale),
        float_to_int8_rn((float)x_val[1].x * tmp_scale),
        float_to_int8_rn((float)x_val[1].y * tmp_scale)
    );

    // calculate sum using reconstructed value
    if constexpr (sum_type == SumType::kPostQuant)
    {
      int32_t local_sum = 0;
      int32_t sum = 0;

      local_sum += static_cast<int32_t>(o_val.x);
      local_sum += static_cast<int32_t>(o_val.y);
      local_sum += static_cast<int32_t>(o_val.z);
      local_sum += static_cast<int32_t>(o_val.w);

      sum = vllm::blockReduceSum(local_sum);
      if (tidx == 0)
      {
        sum_output[bidx] = __float2half_rn(__int2float_rn(sum) / tmp_scale);
      }
    }

    // int8 result
    *reinterpret_cast<char4*>(&output_quant[bidx * hidden_dim + j]) = o_val;
  }
  else
  {
    // fp16 result
    *(float2*)(&output[bidx * hidden_dim + j]) = *(float2*)(&x_val[0]);
  }
}

void layernorm_nobias(torch::Tensor &output,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon) {
  CHECK_CUDA(output);
  CHECK_CUDA(input);
  CHECK_CUDA(weight);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);

  CHECK_DTYPE(output, torch::kFloat16);
  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(weight, torch::kFloat16);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  assert(hidden_size % 128 == 0);

  CHECK_SHAPE(weight, hidden_size);
  CHECK_NUMEL(output, input.numel());

  dim3 grid(num_tokens);
  dim3 block(hidden_size / 4);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  LayernormT2iQuantFuse<false, false, SumType::kNone><<<grid, block, 0, stream>>>(
    reinterpret_cast<half*>(input.data_ptr<at::Half>()),
    reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
    nullptr,
    reinterpret_cast<half*>(output.data_ptr<at::Half>()),
    nullptr,
    nullptr, nullptr,
    epsilon, 0, 0, num_tokens, hidden_size, nullptr);
}

torch::Tensor quant_sum(torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &sum_output, // [tokens]
              torch::Tensor &scaling)
{
  CHECK_CUDA(input);
  CHECK_CUDA(sum_output);
  CHECK_CUDA(scaling);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device


  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sum_output);
  CHECK_CONTIGUOUS(scaling);

  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(sum_output, torch::kFloat16);
  CHECK_DTYPE(scaling, torch::kFloat16);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  assert(hidden_size <= 8192);

  if (hidden_size > 4096) {
    assert(hidden_size % 256 == 0);
  } else {
    assert(hidden_size % 128 == 0);
  }

  CHECK_SHAPE(sum_output, num_tokens);
  CHECK_SHAPE(scaling, num_tokens);

  at::Tensor output = at::empty_like(input, torch::kInt8);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (hidden_size <= 4096) {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 4);
    
    QuantKernel<float2, SumType::kPostQuant><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(scaling.data_ptr<at::Half>()),
      num_tokens, hidden_size);
  }
  else
  {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 8);
    
    QuantKernel<float4, SumType::kPostQuant><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(scaling.data_ptr<at::Half>()),
      num_tokens, hidden_size);
  }

  return output;
}

torch::Tensor quant_sum_bf16(torch::Tensor &input,  // [..., hidden_size]
                        torch::Tensor &sum_output, // [tokens]
                        torch::Tensor &scaling)
{
  CHECK_CUDA(input);
  CHECK_CUDA(sum_output);
  CHECK_CUDA(scaling);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device


  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sum_output);
  CHECK_CONTIGUOUS(scaling);

  CHECK_DTYPE(input, torch::kBFloat16);
  CHECK_DTYPE(sum_output, torch::kBFloat16);
  CHECK_DTYPE(scaling, torch::kBFloat16);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  assert(hidden_size <= 8192);

  if (hidden_size > 4096) {
    assert(hidden_size % 256 == 0);
  } else {
    assert(hidden_size % 128 == 0);
  }

  CHECK_SHAPE(sum_output, num_tokens);
  CHECK_SHAPE(scaling, num_tokens);

  at::Tensor output = at::empty_like(input, torch::kInt8);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (hidden_size <= 4096) {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 4);

    QuantKernelBF16<float2, SumType::kPostQuant><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<__nv_bfloat16*>(sum_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(scaling.data_ptr<at::BFloat16>()),
      num_tokens, hidden_size);
  }
  else {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 8);

    QuantKernelBF16<float4, SumType::kPostQuant><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<__nv_bfloat16*>(sum_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(scaling.data_ptr<at::BFloat16>()),
      num_tokens, hidden_size);
  }

  return output;
}

torch::Tensor quant_sum_static(torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &sum_output, // [tokens]
              torch::Tensor &scaling)
{
  CHECK_CUDA(input);
  CHECK_CUDA(sum_output);
  CHECK_CUDA(scaling);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sum_output);
  CHECK_CONTIGUOUS(scaling);

  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(sum_output, torch::kFloat16);
  CHECK_DTYPE(scaling, torch::kFloat16);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  assert(hidden_size <= 8192);

  if (hidden_size > 4096) {
    assert(hidden_size % 256 == 0);
  } else {
    assert(hidden_size % 128 == 0);
  }

  CHECK_SHAPE(sum_output, num_tokens);
  CHECK_SHAPE(scaling, num_tokens);

  at::Tensor output = at::empty_like(input, torch::kInt8);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (hidden_size <= 4096) {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 4);
    
    QuantKernel<float2, SumType::kPostQuant, false><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(scaling.data_ptr<at::Half>()),
      num_tokens, hidden_size);
  }
  else
  {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 8);
    
    QuantKernel<float4, SumType::kPostQuant, false><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(scaling.data_ptr<at::Half>()),
      num_tokens, hidden_size);
  }

  return output;
}

torch::Tensor gelu_quant_sum(torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &sum_output, // [tokens]
              torch::Tensor &scaling)
{
  CHECK_CUDA(input);
  CHECK_CUDA(sum_output);
  CHECK_CUDA(scaling);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(sum_output);
  CHECK_CONTIGUOUS(scaling);

  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(sum_output, torch::kFloat16);
  CHECK_DTYPE(scaling, torch::kFloat16);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  assert(hidden_size <= 8192);

  if (hidden_size > 4096) {
    assert(hidden_size % 256 == 0);
  } else {
    assert(hidden_size % 128 == 0);
  }

  CHECK_SHAPE(sum_output, num_tokens);
  CHECK_SHAPE(scaling, num_tokens);

  at::Tensor output = at::empty_like(input, torch::kInt8);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (hidden_size <= 4096) {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 4);
    
    GeluQuantFuse<float2, SumType::kPostQuant><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(scaling.data_ptr<at::Half>()),
      num_tokens, hidden_size);
  }
  else
  {
    dim3 grid(num_tokens);
    dim3 block(hidden_size / 8);
    
    GeluQuantFuse<float4, SumType::kPostQuant><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      output.data_ptr<int8_t>(),
      reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(scaling.data_ptr<at::Half>()),
      num_tokens, hidden_size);
  }

  return output;
}

void layernorm_nobias_quant_nosum_fuse(torch::Tensor &output,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &scaling, // [tokens]
              float epsilon) {
  CHECK_CUDA(output);
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(scaling);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(scaling);

  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(weight, torch::kFloat16);
  CHECK_DTYPE(scaling, torch::kFloat16);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  CHECK_SHAPE(weight, hidden_size);
  CHECK_SHAPE(scaling, num_tokens);
  CHECK_NUMEL(output, input.numel());

  assert(hidden_size % 128 == 0);
  dim3 grid(num_tokens);
  dim3 block(hidden_size / 4);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  LayernormT2iQuantFuse<false, true, SumType::kNone><<<grid, block, 0, stream>>>(
    reinterpret_cast<half*>(input.data_ptr<at::Half>()),
    reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
    output.data_ptr<int8_t>(),
    nullptr,
    nullptr,
    nullptr, nullptr,
    epsilon, 0, 0, num_tokens, hidden_size, reinterpret_cast<half*>(scaling.data_ptr<at::Half>()));
}

void layernorm_nobias_quant_sum_fuse(torch::Tensor &output,    // [batch_size * tokens, hidden_size]
              torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &sum_output, // [batch_size * tokens]
              torch::Tensor &scaling, // [batch_size * tokens]
              float epsilon) {
  CHECK_CUDA(output);
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(sum_output);
  CHECK_CUDA(scaling);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(sum_output);
  CHECK_CONTIGUOUS(scaling);

  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(weight, torch::kFloat16);
  CHECK_DTYPE(sum_output, torch::kFloat16);
  CHECK_DTYPE(scaling, torch::kFloat16);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  CHECK_SHAPE(weight, hidden_size);
  CHECK_SHAPE(sum_output, num_tokens);
  CHECK_SHAPE(scaling, num_tokens);
  CHECK_NUMEL(output, input.numel());

  assert(hidden_size % 128 == 0);
  dim3 grid(num_tokens);
  dim3 block(hidden_size / 4);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  LayernormT2iQuantFuse<false, true, SumType::kPostQuant><<<grid, block, 0, stream>>>(
    reinterpret_cast<half*>(input.data_ptr<at::Half>()),
    reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
    output.data_ptr<int8_t>(),
    nullptr,
    reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
    nullptr, nullptr,
    epsilon, 0, 0, num_tokens, hidden_size, reinterpret_cast<half*>(scaling.data_ptr<at::Half>()));
}


void layernorm_nobias_t2i_fuse(torch::Tensor &output,    // [batch_size * tokens, hidden_size]
              torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &shift_msa, // [batch_size, hidden_size]
              torch::Tensor &scale_msa, // [batch_size, hidden_size]
              float epsilon) {
  CHECK_CUDA(output);
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(shift_msa);
  CHECK_CUDA(scale_msa);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_LASTDIM_CONTIGUOUS(shift_msa);
  CHECK_LASTDIM_CONTIGUOUS(scale_msa);

  CHECK_DTYPE(output, torch::kFloat16);
  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(weight, torch::kFloat16);
  CHECK_DTYPE(shift_msa, torch::kFloat16);
  CHECK_DTYPE(scale_msa, torch::kFloat16);

  int batch_size = shift_msa.size(0);
  int hidden_size = shift_msa.size(1);
  int num_tokens = input.size(0) / batch_size;
  int shift_stride = shift_msa.stride(0);
  int scale_stride = scale_msa.stride(0);

  CHECK_SHAPE(input, batch_size * num_tokens, hidden_size);
  CHECK_SHAPE(output, batch_size * num_tokens, hidden_size);
  CHECK_SHAPE(weight, hidden_size);
  CHECK_SHAPE(shift_msa, batch_size, hidden_size);
  CHECK_SHAPE(scale_msa, batch_size, hidden_size);

  assert(hidden_size % 128 == 0);
  dim3 grid(num_tokens * batch_size);
  dim3 block(hidden_size / 4);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  LayernormT2iQuantFuse<true, false><<<grid, block, 0, stream>>>(
    reinterpret_cast<half*>(input.data_ptr<at::Half>()),
    reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
    nullptr,
    reinterpret_cast<half*>(output.data_ptr<at::Half>()),
    nullptr,
    reinterpret_cast<half*>(shift_msa.data_ptr<at::Half>()), reinterpret_cast<half*>(scale_msa.data_ptr<at::Half>()),
    epsilon, shift_stride, scale_stride,
    num_tokens, hidden_size, nullptr);
}


void layernorm_nobias_t2i_quant_sum_fuse(torch::Tensor &output,    // [batch_size * tokens, hidden_size]
              torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &shift_msa, // [batch_size, hidden_size]
              torch::Tensor &scale_msa, // [batch_size, hidden_size]
              torch::Tensor &sum_output, // [batch_size * tokens]
              torch::Tensor &scaling, // [batch_size * tokens]
              float epsilon) {
  CHECK_CUDA(output);
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(shift_msa);
  CHECK_CUDA(scale_msa);
  CHECK_CUDA(sum_output);
  CHECK_CUDA(scaling);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_LASTDIM_CONTIGUOUS(shift_msa);
  CHECK_LASTDIM_CONTIGUOUS(scale_msa);
  CHECK_CONTIGUOUS(sum_output);
  CHECK_CONTIGUOUS(scaling);

  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(weight, torch::kFloat16);
  CHECK_DTYPE(shift_msa, torch::kFloat16);
  CHECK_DTYPE(scale_msa, torch::kFloat16);
  CHECK_DTYPE(sum_output, torch::kFloat16);
  CHECK_DTYPE(scaling, torch::kFloat16);

  int batch_size = shift_msa.size(0);
  int hidden_size = shift_msa.size(1);
  int num_tokens = input.size(0) / batch_size;
  int shift_stride = shift_msa.stride(0);
  int scale_stride = scale_msa.stride(0);

  CHECK_SHAPE(input, batch_size * num_tokens, hidden_size);
  CHECK_SHAPE(output, batch_size * num_tokens, hidden_size);
  CHECK_SHAPE(weight, hidden_size);
  CHECK_SHAPE(shift_msa, batch_size, hidden_size);
  CHECK_SHAPE(scale_msa, batch_size, hidden_size);
  CHECK_SHAPE(sum_output, batch_size * num_tokens);
  CHECK_SHAPE(scaling, batch_size * num_tokens);

  assert(hidden_size % 128 == 0);
  dim3 grid(num_tokens * batch_size);
  dim3 block(hidden_size / 4);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  LayernormT2iQuantFuse<true, true, SumType::kPostQuant><<<grid, block, 0, stream>>>(
    reinterpret_cast<half*>(input.data_ptr<at::Half>()),
    reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
    output.data_ptr<int8_t>(),
    nullptr,
    reinterpret_cast<half*>(sum_output.data_ptr<at::Half>()),
    reinterpret_cast<half*>(shift_msa.data_ptr<at::Half>()), reinterpret_cast<half*>(scale_msa.data_ptr<at::Half>()),
    epsilon, shift_stride, scale_stride,
    num_tokens, hidden_size, reinterpret_cast<half*>(scaling.data_ptr<at::Half>()));
}

std::tuple<at::Tensor, at::Tensor> gate_residual_fuse(torch::Tensor &input,  // [batch_size * tokens, hidden_size]
              torch::Tensor &gate_msa, // [batch_size, hidden_size]
              torch::Tensor &residual // [batch_size * tokens, hidden_size]
              ) {

  CHECK_CUDA(input);
  CHECK_CUDA(gate_msa);
  CHECK_CUDA(residual);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device
  
  CHECK_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(gate_msa);
  CHECK_CONTIGUOUS(residual);

  CHECK_DTYPE(input, torch::kFloat16);
  CHECK_DTYPE(gate_msa, torch::kFloat16);
  CHECK_DTYPE(residual, torch::kFloat16);

  int batch_size = gate_msa.size(0);
  int hidden_size = gate_msa.size(1);
  int num_tokens = input.size(0) / batch_size;
  int gate_stride = gate_msa.stride(0);

  CHECK_SHAPE(input, batch_size * num_tokens, hidden_size);
  CHECK_SHAPE(gate_msa, batch_size, hidden_size);
  CHECK_SHAPE(residual, batch_size * num_tokens, hidden_size);

  torch::Tensor output = at::empty_like(input, torch::kFloat16);
  torch::Tensor intermediate_result = at::empty_like(input, torch::kFloat16);

  assert(hidden_size % 128 == 0);
  dim3 grid(num_tokens * batch_size);
  dim3 block(hidden_size / 4);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  GateResidualQuantFuse<true, false, SumType::kNone><<<grid, block, 0, stream>>>(
    reinterpret_cast<half*>(input.data_ptr<at::Half>()),
    reinterpret_cast<half*>(gate_msa.data_ptr<at::Half>()),
    reinterpret_cast<half*>(residual.data_ptr<at::Half>()),
    nullptr,
    reinterpret_cast<half*>(output.data_ptr<at::Half>()),
    nullptr,
    gate_stride, num_tokens, hidden_size, nullptr,
    reinterpret_cast<half*>(intermediate_result.data_ptr<at::Half>()));

  return std::make_tuple(output, intermediate_result);
}
