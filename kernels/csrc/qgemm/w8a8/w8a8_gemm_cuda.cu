#include "../../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "../../cp_async.cuh"
#include "../../mma.cuh"
#include "../../permuted_smem.cuh"
#include "gemm_utils.cuh"

#define PACK_SIZE_C16 8 // how many elements a load/store 128b instruction can manage for C
#define PACK_SIZE_C32 4

template <uint32_t CTA_M, uint32_t CTA_N, uint32_t CTA_K, uint32_t WARP_M, uint32_t WARP_N, uint32_t CTA_STRIDE, OutputDtype output_dtype=OutputDtype::kFloat16, uint32_t K_STAGE=2, bool has_bias=false, bool weight_asym=false, ScaleMulMode scale_mul_mode=ScaleMulMode::kMode1>
__global__ void GemmInt8SharedRegPipelineV2(const int8_t *__restrict__ A, const int8_t *__restrict__ B,
                              half *__restrict__ C16, int32_t *__restrict__ C32, const half __restrict__ *Bias,
                              const half *__restrict__ scale_A, const half *__restrict__ scale_B,
                              const half *__restrict__ sum_A, const int16_t *__restrict__ zp_B,
                              const int M, const int N, const int K)
{
  static_assert(K_STAGE > 1);

  constexpr uint32_t num_warps_m = CTA_M / WARP_M;
  constexpr uint32_t num_warps_n = CTA_N / WARP_N;
  constexpr uint32_t num_warps = num_warps_m * num_warps_n;
  constexpr uint32_t num_tiles_m = WARP_M / MMA_M;
  constexpr uint32_t num_tiles_n = WARP_N / MMA_N;
  constexpr uint32_t num_tiles_k = CTA_K / MMA_K;

  static_assert(num_tiles_k % 2 == 0);

  constexpr uint32_t AB_SMEM_STRIDE = CTA_K;
  constexpr uint32_t C_SMEM_STRIDE = CTA_N;

  uint32_t blockIdx_m = (blockIdx.z % 2) ? (gridDim.y - blockIdx.y - 1) : blockIdx.y;
  uint32_t blockIdx_n = blockIdx.z * gridDim.x + blockIdx.x;

  if (blockIdx_m >= M / CTA_M || blockIdx_n >= N / CTA_N)
  {
    return;
  }

  extern __shared__ int8_t smem[][AB_SMEM_STRIDE];

  const uint32_t warp_id = get_warp_id();
  const uint32_t lane_id = get_lane_id();

  // RC holds the fragment of C
  int32_t RC[num_tiles_m][num_tiles_n][8];

  // initialize RC
#pragma unroll
  for (uint32_t i = 0; i < num_tiles_m; ++i)
  {
#pragma unroll
    for (uint32_t j = 0; j < num_tiles_n; ++j)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; ++k)
      {
        RC[i][j][k] = 0;
      }
    }
  }

  constexpr uint32_t B_smem_idx_off = CTA_M;
  constexpr uint32_t smem_stage_off = CTA_M + CTA_N;

  constexpr SwizzleMode swizzle_mode_AB = (AB_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE> current_smem_A(smem);
  smem_t<swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE> current_smem_B(smem + B_smem_idx_off);

  constexpr SwizzleMode swizzle_mode_C16 = (C_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_C16, C_SMEM_STRIDE / PACK_SIZE_C16> smem_C16(smem);
  constexpr SwizzleMode swizzle_mode_C32 = (C_SMEM_STRIDE == 16) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_C32, C_SMEM_STRIDE / PACK_SIZE_C32> smem_C32(smem);

  // A_warp_base_ptr and B_warp_base_ptr are used to load data from global memory to shared memory
  // each warp loads a few rows for A
  const int8_t *A_warp_base_ptr = A + blockIdx_m * CTA_M * K + CTA_M / num_warps * warp_id * K;
  const int8_t *B_warp_base_ptr = B + blockIdx_n * CTA_N * K + CTA_N / num_warps * warp_id * K;
  half *C16_warp_base_ptr = C16 + blockIdx_m * CTA_M * N + CTA_M / num_warps * warp_id * N + blockIdx_n * CTA_N;
  int32_t *C32_warp_base_ptr = C32 + blockIdx_m * CTA_M * N + CTA_M / num_warps * warp_id * N + blockIdx_n * CTA_N;

  constexpr uint32_t global_to_shared_line_lanes = (AB_SMEM_STRIDE == 64) ? 4 : 8; // when loading from global to shared memory, how many lanes are used to load a line
  constexpr uint32_t global_to_shared_copy_lines_per_warp = (AB_SMEM_STRIDE == 64) ? 8 : 4; // how many lines are copied per warp per iteration

  constexpr uint32_t global_to_shared_line_lanes_C16 = (C_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_C16 = (C_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_C32 = (C_SMEM_STRIDE == 16) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_C32 = (C_SMEM_STRIDE == 16) ? 8 : 4;

  constexpr uint32_t A_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE);
  constexpr uint32_t B_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE);
  constexpr uint32_t A_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp);
  constexpr uint32_t B_smem_iters_col = CTA_N / (num_warps * global_to_shared_copy_lines_per_warp);

  constexpr uint32_t C16_smem_iters_row = C_SMEM_STRIDE / (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
  constexpr uint32_t C16_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp_C16);
  constexpr uint32_t C32_smem_iters_row = C_SMEM_STRIDE / (global_to_shared_line_lanes_C32 * PACK_SIZE_C32);
  constexpr uint32_t C32_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp_C32);


  // store idx is used to store data to shared memory
  uint32_t smem_store_idx = K_STAGE - 1, smem_store_off = 0;
  // load idx is used to load data from shared memory to registers
  uint32_t smem_load_idx = 0, smem_load_off = 0;


  // prefetch K_STAGE stages of data
#pragma unroll
  for (uint32_t stage = 0; stage < K_STAGE; stage++)
  {
    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_store_off);
    current_smem_B.set_base(smem + smem_store_off + B_smem_idx_off);

    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      A_warp_base_ptr + stage * CTA_K, K, current_smem_A);
    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      B_warp_base_ptr + stage * CTA_K, K, current_smem_B);

    cp_async::commit_group();
  }

  // ensure stage 0 is ready
  cp_async::wait_group<K_STAGE - 1>();
  __syncthreads();

  // store_idx is used to store data to register
  uint32_t reg_store_idx = 0;
  // load_idx is used to load data from register to tensor core
  uint32_t reg_load_idx = 1;

  uint32_t RA[2][num_tiles_m][4];
  uint32_t RB[2][num_tiles_n][4];


  current_smem_A.set_base(smem + smem_load_off);
  current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

  share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_A, RA[reg_store_idx], 0);
  share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_B, RB[reg_store_idx], 0);

  reg_store_idx ^= 1; // reg_store_idx is 1 here
  reg_load_idx ^= 1; // reg_load_idx is 0 here

  share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_A, RA[reg_store_idx], 2);
  share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_B, RB[reg_store_idx], 2);

  tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

#pragma unroll
  for (uint32_t k = 2; k < num_tiles_k; k += 2)
  {
    reg_store_idx ^= 1; // reg_store_idx is 0 here
    reg_load_idx ^= 1; // reg_load_idx is 1 here

    share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_A, RA[reg_store_idx], 2 * k);
    share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_B, RB[reg_store_idx], 2 * k);

    tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

    reg_store_idx ^= 1; // reg_store_idx is 1 here
    reg_load_idx ^= 1; // reg_load_idx is 0 here

    share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_A, RA[reg_store_idx], 2 * k + 2);
    share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_B, RB[reg_store_idx], 2 * k + 2);

    tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
  }

  __syncthreads();

#pragma unroll
  for (uint32_t offset_K = K_STAGE * CTA_K; offset_K < K; offset_K += CTA_K)
  {
    // store data for shared memory from global memory
    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_store_off);
    current_smem_B.set_base(smem + smem_store_off + B_smem_idx_off);

    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      A_warp_base_ptr + offset_K, K, current_smem_A);
    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      B_warp_base_ptr + offset_K, K, current_smem_B);

    cp_async::commit_group();

    cp_async::wait_group<K_STAGE - 1>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }

    __syncthreads();
  }

  static_assert(K_STAGE <= 5);

  if constexpr (K_STAGE >= 5)
  {
    cp_async::wait_group<3>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }

  if constexpr (K_STAGE >= 4)
  {
    cp_async::wait_group<2>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }

  if constexpr (K_STAGE >= 3)
  {
    cp_async::wait_group<1>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }


  if constexpr (K_STAGE >= 2)
  {
    cp_async::wait_group<0>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }

  if constexpr (K_STAGE >= 1)
  {
    __syncthreads();
    reg_store_idx ^= 1; // reg_store_idx is 0 here
    reg_load_idx ^= 1; // reg_load_idx is 1 here

    tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
  }

  // fp16 output
  if constexpr (output_dtype == OutputDtype::kFloat16)
  {
    // not well optimized, but this part is not bottleneck
    const half *scale_A_warp_ptr = scale_A + blockIdx_m * CTA_M + get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M;
    const half *scale_B_warp_ptr = scale_B + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;
    const int16_t *zp_B_warp_ptr = zp_B + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;
    const half *sum_a_warp_ptr = sum_A + blockIdx_m * CTA_M + get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M;
    const half *bias_warp_ptr = Bias + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;

    float a_scale = 1.0f;
    float2 b_scale = {1.0f, 1.0f};
    float a_sum = 0.0f;
    short2 zp_b = {0, 0};
    float2 bias = {0.0f, 0.0f};
    float2 psums = {0.0f, 0.0f};
#pragma unroll
    for (uint32_t i = 0; i < num_tiles_m; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < num_tiles_n; j++)
      {
        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4));
        b_scale = __half22float2(*reinterpret_cast<const half2*>(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4)));
        psums = make_float2(__int2float_rn(RC[i][j][0]), __int2float_rn(RC[i][j][1]));
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1)
        {
          psums.x = psums.x * a_scale * b_scale.x;
          psums.y = psums.y * a_scale * b_scale.y;
        }
        else if constexpr (scale_mul_mode == ScaleMulMode::kMode2)
        {
          psums.x *= a_scale * b_scale.x;
          psums.y *= a_scale * b_scale.y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __half2float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4));
          zp_b = *reinterpret_cast<const short2*>(zp_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4));
          psums.x = psums.x + a_sum * static_cast<float>(zp_b.x) * b_scale.x;
          psums.y = psums.y + a_sum * static_cast<float>(zp_b.y) * b_scale.y;
        }
        if constexpr (has_bias)
        {
          bias = __half22float2(*reinterpret_cast<const half2*>(bias_warp_ptr + j * MMA_N + 2 * (lane_id % 4)));
          psums.x += bias.x;
          psums.y += bias.y;
        }
        ((half2*)RC[i][j])[0] = __float22half2_rn(psums);

        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4 + 8));
        psums = make_float2(__int2float_rn(RC[i][j][2]), __int2float_rn(RC[i][j][3]));
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1)
        {
          psums.x = psums.x * a_scale * b_scale.x;
          psums.y = psums.y * a_scale * b_scale.y;
        }
        else if constexpr (scale_mul_mode == ScaleMulMode::kMode2)
        {
          psums.x *= a_scale * b_scale.x;
          psums.y *= a_scale * b_scale.y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __half2float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4 + 8));
          psums.x = psums.x + a_sum * static_cast<float>(zp_b.x) * b_scale.x;
          psums.y = psums.y + a_sum * static_cast<float>(zp_b.y) * b_scale.y;
        }
        if constexpr (has_bias)
        {
          psums.x += bias.x;
          psums.y += bias.y;
        }
        ((half2*)RC[i][j])[2] = __float22half2_rn(psums);

        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4));
        b_scale = __half22float2(*reinterpret_cast<const half2*>(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 8));
        psums = make_float2(__int2float_rn(RC[i][j][4]), __int2float_rn(RC[i][j][5]));
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1)
        {
          psums.x = psums.x * a_scale * b_scale.x;
          psums.y = psums.y * a_scale * b_scale.y;
        }
        else if constexpr (scale_mul_mode == ScaleMulMode::kMode2)
        {
          psums.x *= a_scale * b_scale.x;
          psums.y *= a_scale * b_scale.y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __half2float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4));
          zp_b = *reinterpret_cast<const short2*>(zp_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 8);
          psums.x = psums.x + a_sum * static_cast<float>(zp_b.x) * b_scale.x;
          psums.y = psums.y + a_sum * static_cast<float>(zp_b.y) * b_scale.y;
        }
        if constexpr (has_bias)
        {
          bias = __half22float2(*reinterpret_cast<const half2*>(bias_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 8));
          psums.x += bias.x;
          psums.y += bias.y;
        }
        ((half2*)RC[i][j])[4] = __float22half2_rn(psums);

        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4 + 8));
        psums = make_float2(__int2float_rn(RC[i][j][6]), __int2float_rn(RC[i][j][7]));
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1)
        {
          psums.x = psums.x * a_scale * b_scale.x;
          psums.y = psums.y * a_scale * b_scale.y;
        }
        else if constexpr (scale_mul_mode == ScaleMulMode::kMode2)
        {
          psums.x *= a_scale * b_scale.x;
          psums.y *= a_scale * b_scale.y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __half2float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4 + 8));
          psums.x = psums.x + a_sum * static_cast<float>(zp_b.x) * b_scale.x;
          psums.y = psums.y + a_sum * static_cast<float>(zp_b.y) * b_scale.y;
        }
        if constexpr (has_bias)
        {
          psums.x += bias.x;
          psums.y += bias.y;
        }
        ((half2*)RC[i][j])[6] = __float22half2_rn(psums);
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < num_tiles_m; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < num_tiles_n; j++)
      {
        uint32_t offset_C1 = smem_C16.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N / PACK_SIZE_C16));

        ((int32_t*)(smem_C16.base + offset_C1))[lane_id % 4] = RC[i][j][0];
        ((int32_t*)(smem_C16.base + offset_C1 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][2];

        uint32_t offset_C2 = smem_C16.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N / PACK_SIZE_C16) + 1);

        ((int32_t*)(smem_C16.base + offset_C2))[lane_id % 4] = RC[i][j][4];
        ((int32_t*)(smem_C16.base + offset_C2 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][6];
      }
    }

    __syncthreads();

    half *C_lane_ptr = C16_warp_base_ptr + lane_id / global_to_shared_line_lanes_C16 * N + lane_id % global_to_shared_line_lanes_C16 * PACK_SIZE_C16;
    uint32_t offset_C = smem_C16.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_C16 * C16_smem_iters_col + lane_id / global_to_shared_line_lanes_C16, lane_id % global_to_shared_line_lanes_C16);

#pragma unroll
    for (uint32_t i = 0; i < C16_smem_iters_col; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < C16_smem_iters_row; j++)
      {
        smem_C16.store_128b(offset_C, C_lane_ptr);
        C_lane_ptr += (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
        offset_C = smem_C16.advance_offset_by_column<global_to_shared_line_lanes_C16>(offset_C);
      }

      offset_C = smem_C16.advance_offset_by_row<global_to_shared_copy_lines_per_warp_C16>(offset_C - (C16_smem_iters_row * global_to_shared_line_lanes_C16));
      C_lane_ptr += ((global_to_shared_copy_lines_per_warp_C16 * N) - (C16_smem_iters_row * global_to_shared_line_lanes_C16 * PACK_SIZE_C16));
    }

  }

  // Int32 output
  if constexpr (output_dtype == OutputDtype::kInt32)
  {

    uint32_t lane_offset_smem_C = 2 * ((lane_id % 4) % 2);
#pragma unroll
    for (uint32_t i = 0; i < num_tiles_m; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < num_tiles_n; j++)
      {
        uint32_t offset_C1 = smem_C32.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C32) + j * (MMA_N / PACK_SIZE_C32) + (lane_id % 4) / 2);

        ((int32_t*)(smem_C32.base + offset_C1))[lane_offset_smem_C] = RC[i][j][0];
        ((int32_t*)(smem_C32.base + offset_C1))[lane_offset_smem_C + 1] = RC[i][j][1];

        ((int32_t*)(smem_C32.base + offset_C1 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C32)))[lane_offset_smem_C] = RC[i][j][2];
        ((int32_t*)(smem_C32.base + offset_C1 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C32)))[lane_offset_smem_C + 1] = RC[i][j][3];

        uint32_t offset_C2 = smem_C32.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N / PACK_SIZE_C32 + j * MMA_N / PACK_SIZE_C32 + (lane_id % 4) / 2 + 2);

        ((int32_t*)(smem_C32.base + offset_C2))[lane_offset_smem_C] = RC[i][j][4];
        ((int32_t*)(smem_C32.base + offset_C2))[lane_offset_smem_C + 1] = RC[i][j][5];

        ((int32_t*)(smem_C32.base + offset_C2 + (8 * C_SMEM_STRIDE / PACK_SIZE_C32)))[lane_offset_smem_C] = RC[i][j][6];
        ((int32_t*)(smem_C32.base + offset_C2 + (8 * C_SMEM_STRIDE / PACK_SIZE_C32)))[lane_offset_smem_C + 1] = RC[i][j][7];
      }
    }
    __syncthreads();

    int32_t *C_lane_ptr = C32_warp_base_ptr + lane_id / global_to_shared_line_lanes_C32 * N + lane_id % global_to_shared_line_lanes_C32 * PACK_SIZE_C32;
    uint32_t offset_C = smem_C32.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_C32 * C32_smem_iters_col + lane_id / global_to_shared_line_lanes_C32, lane_id % global_to_shared_line_lanes_C32);

#pragma unroll
    for (uint32_t i = 0; i < C32_smem_iters_col; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < C32_smem_iters_row; j++)
      {
        smem_C32.store_128b(offset_C, C_lane_ptr);
        C_lane_ptr += (global_to_shared_line_lanes_C32 * PACK_SIZE_C32);
        offset_C = smem_C32.advance_offset_by_column<global_to_shared_line_lanes_C32>(offset_C);
      }

      offset_C = smem_C32.advance_offset_by_row<global_to_shared_copy_lines_per_warp_C32>(offset_C - (C32_smem_iters_row * global_to_shared_line_lanes_C32));
      C_lane_ptr += ((global_to_shared_copy_lines_per_warp_C32 * N) - (C32_smem_iters_row * global_to_shared_line_lanes_C32 * PACK_SIZE_C32));
    }

  }

}

// only support bfloat16 output, no other dtype like float16 kernel
template <uint32_t CTA_M, uint32_t CTA_N, uint32_t CTA_K,
          uint32_t WARP_M, uint32_t WARP_N, uint32_t CTA_STRIDE,
          uint32_t K_STAGE = 2,
          bool has_bias = false, bool weight_asym = false,
          ScaleMulMode scale_mul_mode = ScaleMulMode::kMode1>
__global__ void GemmInt8SharedRegPipelineBF16(
    const int8_t *__restrict__ A,
    const int8_t *__restrict__ B,
    __nv_bfloat16 *__restrict__ C, 
    const __nv_bfloat16 *__restrict__ Bias,
    const __nv_bfloat16 *__restrict__ scale_A,
    const __nv_bfloat16 *__restrict__ scale_B,
    const __nv_bfloat16 *__restrict__ sum_A,
    const int16_t *__restrict__ zp_B,
    const int M, const int N, const int K)
{
  static_assert(K_STAGE > 1);

  constexpr uint32_t num_warps_m = CTA_M / WARP_M;
  constexpr uint32_t num_warps_n = CTA_N / WARP_N;
  constexpr uint32_t num_warps = num_warps_m * num_warps_n;
  constexpr uint32_t num_tiles_m = WARP_M / MMA_M;
  constexpr uint32_t num_tiles_n = WARP_N / MMA_N;
  constexpr uint32_t num_tiles_k = CTA_K / MMA_K;

  static_assert(num_tiles_k % 2 == 0);

  constexpr uint32_t AB_SMEM_STRIDE = CTA_K;
  constexpr uint32_t C_SMEM_STRIDE = CTA_N;

  uint32_t blockIdx_m = (blockIdx.z % 2) ? (gridDim.y - blockIdx.y - 1) : blockIdx.y;
  uint32_t blockIdx_n = blockIdx.z * gridDim.x + blockIdx.x;

  if (blockIdx_m >= M / CTA_M || blockIdx_n >= N / CTA_N)
  {
    return;
  }

  extern __shared__ int8_t smem[][AB_SMEM_STRIDE];

  const uint32_t warp_id = get_warp_id();
  const uint32_t lane_id = get_lane_id();

  // RC holds the fragment of C
  int32_t RC[num_tiles_m][num_tiles_n][8];

  // initialize RC
#pragma unroll
  for (uint32_t i = 0; i < num_tiles_m; ++i)
  {
#pragma unroll
    for (uint32_t j = 0; j < num_tiles_n; ++j)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; ++k)
      {
        RC[i][j][k] = 0;
      }
    }
  }

  constexpr uint32_t B_smem_idx_off = CTA_M;
  constexpr uint32_t smem_stage_off = CTA_M + CTA_N;

  constexpr SwizzleMode swizzle_mode_AB = (AB_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE> current_smem_A(smem);
  smem_t<swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE> current_smem_B(smem + B_smem_idx_off);

  constexpr SwizzleMode swizzle_mode_C16 = (C_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_C16, C_SMEM_STRIDE / PACK_SIZE_C16> smem_C16(smem);
  constexpr SwizzleMode swizzle_mode_C32 = (C_SMEM_STRIDE == 16) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_C32, C_SMEM_STRIDE / PACK_SIZE_C32> smem_C32(smem);

  // A_warp_base_ptr and B_warp_base_ptr are used to load data from global memory to shared memory
  // each warp loads a few rows for A
  const int8_t *A_warp_base_ptr = A + blockIdx_m * CTA_M * K + CTA_M / num_warps * warp_id * K;
  const int8_t *B_warp_base_ptr = B + blockIdx_n * CTA_N * K + CTA_N / num_warps * warp_id * K;
  __nv_bfloat16 *C_warp_base_ptr = C + + blockIdx_m * CTA_M * N + CTA_M / num_warps * warp_id * N + blockIdx_n * CTA_N;
//   int32_t *C32_warp_base_ptr = C32 + blockIdx_m * CTA_M * N + CTA_M / num_warps * warp_id * N + blockIdx_n * CTA_N;

  constexpr uint32_t global_to_shared_line_lanes = (AB_SMEM_STRIDE == 64) ? 4 : 8; // when loading from global to shared memory, how many lanes are used to load a line
  constexpr uint32_t global_to_shared_copy_lines_per_warp = (AB_SMEM_STRIDE == 64) ? 8 : 4; // how many lines are copied per warp per iteration

  constexpr uint32_t global_to_shared_line_lanes_C16 = (C_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_C16 = (C_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_C32 = (C_SMEM_STRIDE == 16) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_C32 = (C_SMEM_STRIDE == 16) ? 8 : 4;

  constexpr uint32_t A_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE);
  constexpr uint32_t B_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE);
  constexpr uint32_t A_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp);
  constexpr uint32_t B_smem_iters_col = CTA_N / (num_warps * global_to_shared_copy_lines_per_warp);

  constexpr uint32_t C16_smem_iters_row = C_SMEM_STRIDE / (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
  constexpr uint32_t C16_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp_C16);
  constexpr uint32_t C32_smem_iters_row = C_SMEM_STRIDE / (global_to_shared_line_lanes_C32 * PACK_SIZE_C32);
  constexpr uint32_t C32_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp_C32);


  // store idx is used to store data to shared memory
  uint32_t smem_store_idx = K_STAGE - 1, smem_store_off = 0;
  // load idx is used to load data from shared memory to registers
  uint32_t smem_load_idx = 0, smem_load_off = 0;


  // prefetch K_STAGE stages of data
#pragma unroll
  for (uint32_t stage = 0; stage < K_STAGE; stage++)
  {
    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_store_off);
    current_smem_B.set_base(smem + smem_store_off + B_smem_idx_off);

    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      A_warp_base_ptr + stage * CTA_K, K, current_smem_A);
    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      B_warp_base_ptr + stage * CTA_K, K, current_smem_B);

    cp_async::commit_group();
  }

  // ensure stage 0 is ready
  cp_async::wait_group<K_STAGE - 1>();
  __syncthreads();

  // store_idx is used to store data to register
  uint32_t reg_store_idx = 0;
  // load_idx is used to load data from register to tensor core
  uint32_t reg_load_idx = 1;

  uint32_t RA[2][num_tiles_m][4];
  uint32_t RB[2][num_tiles_n][4];


  current_smem_A.set_base(smem + smem_load_off);
  current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

  share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_A, RA[reg_store_idx], 0);
  share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_B, RB[reg_store_idx], 0);

  reg_store_idx ^= 1; // reg_store_idx is 1 here
  reg_load_idx ^= 1; // reg_load_idx is 0 here

  share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_A, RA[reg_store_idx], 2);
  share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
    current_smem_B, RB[reg_store_idx], 2);

  tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

#pragma unroll
  for (uint32_t k = 2; k < num_tiles_k; k += 2)
  {
    reg_store_idx ^= 1; // reg_store_idx is 0 here
    reg_load_idx ^= 1; // reg_load_idx is 1 here

    share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_A, RA[reg_store_idx], 2 * k);
    share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_B, RB[reg_store_idx], 2 * k);

    tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

    reg_store_idx ^= 1; // reg_store_idx is 1 here
    reg_load_idx ^= 1; // reg_load_idx is 0 here

    share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_A, RA[reg_store_idx], 2 * k + 2);
    share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      current_smem_B, RB[reg_store_idx], 2 * k + 2);

    tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
  }

  __syncthreads();

#pragma unroll
  for (uint32_t offset_K = K_STAGE * CTA_K; offset_K < K; offset_K += CTA_K)
  {
    // store data for shared memory from global memory
    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_store_off);
    current_smem_B.set_base(smem + smem_store_off + B_smem_idx_off);

    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      A_warp_base_ptr + offset_K, K, current_smem_A);
    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
      B_warp_base_ptr + offset_K, K, current_smem_B);

    cp_async::commit_group();

    cp_async::wait_group<K_STAGE - 1>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }

    __syncthreads();
  }

  static_assert(K_STAGE <= 5);

  if constexpr (K_STAGE >= 5)
  {
    cp_async::wait_group<3>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }

  if constexpr (K_STAGE >= 4)
  {
    cp_async::wait_group<2>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }

  if constexpr (K_STAGE >= 3)
  {
    cp_async::wait_group<1>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }


  if constexpr (K_STAGE >= 2)
  {
    cp_async::wait_group<0>();
    __syncthreads();

    smem_load_idx = (smem_load_idx + 1) % K_STAGE;
    smem_load_off = smem_load_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_load_off);
    current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 2)
    {
      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

      reg_store_idx ^= 1; // reg_store_idx is 1 here
      reg_load_idx ^= 1; // reg_load_idx is 0 here

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_A, RA[reg_store_idx], 2 * k + 2);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
        current_smem_B, RB[reg_store_idx], 2 * k + 2);

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }

  if constexpr (K_STAGE >= 1)
  {
    __syncthreads();
    reg_store_idx ^= 1; // reg_store_idx is 0 here
    reg_load_idx ^= 1; // reg_load_idx is 1 here

    tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
  }

    // not well optimized, but this part is not bottleneck
    const __nv_bfloat16 *scale_A_warp_ptr = scale_A + blockIdx_m * CTA_M + get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M;
    const __nv_bfloat16 *scale_B_warp_ptr = scale_B + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;
    const int16_t *zp_B_warp_ptr = zp_B + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;
    const __nv_bfloat16 *sum_a_warp_ptr = sum_A + blockIdx_m * CTA_M + get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M;
    const __nv_bfloat16 *bias_warp_ptr = Bias + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;

    float a_scale = 1.0f;
    float b_scale_x = 1.0f, b_scale_y = 1.0f;
    float a_sum = 0.0f;
    short2 zp_b = {0, 0};
    float bias_x = 0.0f, bias_y = 0.0f;
    float psum_x = 0.0f, psum_y = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < num_tiles_m; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < num_tiles_n; j++)
      {
        a_scale = __bfloat162float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4));
        b_scale_x = __bfloat162float(*(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4)));
        b_scale_y = __bfloat162float(*(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 1));
        psum_x = static_cast<float>(RC[i][j][0]);
        psum_y = static_cast<float>(RC[i][j][1]);
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1 || scale_mul_mode == ScaleMulMode::kMode2)
        {
          psum_x *= a_scale * b_scale_x;
          psum_y *= a_scale * b_scale_y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __bfloat162float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4));
          zp_b = *reinterpret_cast<const short2*>(zp_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4));
          psum_x += a_sum * static_cast<float>(zp_b.x) * b_scale_x;
          psum_y += a_sum * static_cast<float>(zp_b.y) * b_scale_y;
        }
        if constexpr (has_bias)
        {
          bias_x = __bfloat162float(*(bias_warp_ptr + j * MMA_N + 2 * (lane_id % 4)));
          bias_y = __bfloat162float(*(bias_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 1));
          psum_x += bias_x;
          psum_y += bias_y;
        }
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][0])[0] = __float2bfloat16(psum_x);
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][0])[1] = __float2bfloat16(psum_y);

        a_scale = __bfloat162float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4 + 8));
        psum_x = static_cast<float>(RC[i][j][2]);
        psum_y = static_cast<float>(RC[i][j][3]);
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1 || scale_mul_mode == ScaleMulMode::kMode2)
        {
          psum_x *= a_scale * b_scale_x;
          psum_y *= a_scale * b_scale_y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __bfloat162float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4 + 8));
          psum_x += a_sum * static_cast<float>(zp_b.x) * b_scale_x;
          psum_y += a_sum * static_cast<float>(zp_b.y) * b_scale_y;
        }
        if constexpr (has_bias)
        {
          psum_x += bias_x;
          psum_y += bias_y;
        }
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][2])[0] = __float2bfloat16(psum_x);
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][2])[1] = __float2bfloat16(psum_y);

        a_scale = __bfloat162float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4));
        b_scale_x = __bfloat162float(*(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 8));
        b_scale_y = __bfloat162float(*(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 9));
        psum_x = static_cast<float>(RC[i][j][4]);
        psum_y = static_cast<float>(RC[i][j][5]);
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1 || scale_mul_mode == ScaleMulMode::kMode2)
        {
          psum_x *= a_scale * b_scale_x;
          psum_y *= a_scale * b_scale_y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __bfloat162float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4));
          zp_b = *reinterpret_cast<const short2*>(zp_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 8);
          psum_x += a_sum * static_cast<float>(zp_b.x) * b_scale_x;
          psum_y += a_sum * static_cast<float>(zp_b.y) * b_scale_y;
        }
        if constexpr (has_bias)
        {
          bias_x = __bfloat162float(*(bias_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 8));
          bias_y = __bfloat162float(*(bias_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 9));
          psum_x += bias_x;
          psum_y += bias_y;
        }
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][4])[0] = __float2bfloat16(psum_x);
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][4])[1] = __float2bfloat16(psum_y);

        a_scale = __bfloat162float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4 + 8));
        psum_x = static_cast<float>(RC[i][j][6]);
        psum_y = static_cast<float>(RC[i][j][7]);
        if constexpr (scale_mul_mode == ScaleMulMode::kMode1 || scale_mul_mode == ScaleMulMode::kMode2)
        {
          psum_x *= a_scale * b_scale_x;
          psum_y *= a_scale * b_scale_y;
        }
        if constexpr (weight_asym)
        {
          a_sum = __bfloat162float(*(sum_a_warp_ptr + i * MMA_M + lane_id / 4 + 8));
          psum_x += a_sum * static_cast<float>(zp_b.x) * b_scale_x;
          psum_y += a_sum * static_cast<float>(zp_b.y) * b_scale_y;
        }
        if constexpr (has_bias)
        {
          psum_x += bias_x;
          psum_y += bias_y;
        }
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][6])[0] = __float2bfloat16(psum_x);
        reinterpret_cast<__nv_bfloat16*>(&RC[i][j][6])[1] = __float2bfloat16(psum_y);
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < num_tiles_m; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < num_tiles_n; j++)
      {
        uint32_t offset_C1 = smem_C16.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N / PACK_SIZE_C16));

        ((int32_t*)(smem_C16.base + offset_C1))[lane_id % 4] = RC[i][j][0];
        ((int32_t*)(smem_C16.base + offset_C1 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][2];

        uint32_t offset_C2 = smem_C16.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N / PACK_SIZE_C16) + 1);

        ((int32_t*)(smem_C16.base + offset_C2))[lane_id % 4] = RC[i][j][4];
        ((int32_t*)(smem_C16.base + offset_C2 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][6];
      }
    }

    __syncthreads();

    __nv_bfloat16 *C_lane_ptr = C_warp_base_ptr + lane_id / global_to_shared_line_lanes_C16 * N + lane_id % global_to_shared_line_lanes_C16 * PACK_SIZE_C16;
    uint32_t offset_C = smem_C16.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_C16 * C16_smem_iters_col + lane_id / global_to_shared_line_lanes_C16, lane_id % global_to_shared_line_lanes_C16);

#pragma unroll
    for (uint32_t i = 0; i < C16_smem_iters_col; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < C16_smem_iters_row; j++)
      {
        smem_C16.store_128b(offset_C, C_lane_ptr);
        C_lane_ptr += (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
        offset_C = smem_C16.advance_offset_by_column<global_to_shared_line_lanes_C16>(offset_C);
      }

      offset_C = smem_C16.advance_offset_by_row<global_to_shared_copy_lines_per_warp_C16>(offset_C - (C16_smem_iters_row * global_to_shared_line_lanes_C16));
      C_lane_ptr += ((global_to_shared_copy_lines_per_warp_C16 * N) - (C16_smem_iters_row * global_to_shared_line_lanes_C16 * PACK_SIZE_C16));
    }


}

torch::Tensor w8a8_of16_bias_weight_asym(torch::Tensor input,
                      torch::Tensor weight,
                      torch::Tensor bias,
                      torch::Tensor scale_input,
                      torch::Tensor scale_weight,
                      torch::Tensor sum_input,
                      torch::Tensor zp_weight)
{
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  CHECK_CUDA(scale_input);
  CHECK_CUDA(scale_weight);
  CHECK_CUDA(sum_input);
  CHECK_CUDA(zp_weight);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);
  CHECK_CONTIGUOUS(scale_input);
  CHECK_CONTIGUOUS(scale_weight);
  CHECK_CONTIGUOUS(sum_input);
  CHECK_CONTIGUOUS(zp_weight);

  CHECK_DTYPE(input, torch::kInt8);
  CHECK_DTYPE(weight, torch::kInt8);
  CHECK_DTYPE(bias, torch::kHalf);
  CHECK_DTYPE(scale_input, torch::kHalf);
  CHECK_DTYPE(scale_weight, torch::kHalf);
  CHECK_DTYPE(sum_input, torch::kHalf);
  CHECK_DTYPE(zp_weight, torch::kInt16);

  const int M = input.size(0);
  const int N = weight.size(0);
  const int K = input.size(1);

  CHECK_SHAPE(input, M, K);
  CHECK_SHAPE(weight, N, K);
  CHECK_SHAPE(bias, N);
  CHECK_SHAPE(scale_input, M);
  CHECK_SHAPE(scale_weight, N);
  CHECK_SHAPE(sum_input, M);
  CHECK_SHAPE(zp_weight, N);

  at::Tensor output = torch::empty({input.size(0), weight.size(0)}, input.options().dtype(torch::kHalf));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int CTA_M = 128;
  const int CTA_N = 128;
  const int CTA_K = 64;
  constexpr int WARP_M = 128;
  constexpr int WARP_N = 32;
  constexpr int CTA_STRIDE = 1;
  constexpr int K_STAGE = 3;

  if ((M % CTA_M != 0) || (N % CTA_N != 0) || (K % CTA_K != 0)){
    printf("(%d,%d)x(%d,%d) is require, but (%d,%d,%d) must %% (%d,%d,%d)==0", M,K,K,N,M,N,K,CTA_M,CTA_N,CTA_K);
  }

  assert(M % CTA_M == 0);
  assert(N % CTA_N == 0);
  assert(K % CTA_K == 0);

  size_t smem_max = std::max((CTA_M * CTA_K + CTA_N * CTA_K) * sizeof(int8_t) * K_STAGE, CTA_M * CTA_N * sizeof(half));

  auto kernel_func = GemmInt8SharedRegPipelineV2<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, CTA_STRIDE, OutputDtype::kFloat16, K_STAGE, true, true>;

  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(CTA_STRIDE, M / CTA_M, div_ceil(N / CTA_N, CTA_STRIDE));
  dim3 block(32, (CTA_M / WARP_M) * (CTA_N / WARP_N));

  kernel_func<<<grid, block, smem_max, stream>>>(
    input.data_ptr<int8_t>(),
    weight.data_ptr<int8_t>(),
    reinterpret_cast<half*>(output.data_ptr()),
    nullptr,
    reinterpret_cast<half*>(bias.data_ptr()),
    reinterpret_cast<half*>(scale_input.data_ptr()),
    reinterpret_cast<half*>(scale_weight.data_ptr()),
    reinterpret_cast<half*>(sum_input.data_ptr()),
    zp_weight.data_ptr<int16_t>(),
    M, N, K);

  return output;
}

torch::Tensor w8a8_bf16_bias_weight_asym(torch::Tensor input,
                                         torch::Tensor weight,
                                         torch::Tensor bias,
                                         torch::Tensor scale_input,
                                         torch::Tensor scale_weight,
                                         torch::Tensor sum_input,
                                         torch::Tensor zp_weight)
{
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  CHECK_CUDA(scale_input);
  CHECK_CUDA(scale_weight);
  CHECK_CUDA(sum_input);
  CHECK_CUDA(zp_weight);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);
  CHECK_CONTIGUOUS(scale_input);
  CHECK_CONTIGUOUS(scale_weight);
  CHECK_CONTIGUOUS(sum_input);
  CHECK_CONTIGUOUS(zp_weight);

  CHECK_DTYPE(input, torch::kInt8);
  CHECK_DTYPE(weight, torch::kInt8);
  CHECK_DTYPE(bias, torch::kBFloat16);
  CHECK_DTYPE(scale_input, torch::kBFloat16);
  CHECK_DTYPE(scale_weight, torch::kBFloat16);
  CHECK_DTYPE(sum_input, torch::kBFloat16);
  CHECK_DTYPE(zp_weight, torch::kInt16);

  const int M = input.size(0);
  const int N = weight.size(0);
  const int K = input.size(1);

  CHECK_SHAPE(input, M, K);
  CHECK_SHAPE(weight, N, K);
  CHECK_SHAPE(bias, N);
  CHECK_SHAPE(scale_input, M);
  CHECK_SHAPE(scale_weight, N);
  CHECK_SHAPE(sum_input, M);
  CHECK_SHAPE(zp_weight, N);

  at::Tensor output = torch::empty({M, N}, input.options().dtype(torch::kBFloat16));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int CTA_M = 128;
  constexpr int CTA_N = 128;
  constexpr int CTA_K = 64;
  constexpr int WARP_M = 128;
  constexpr int WARP_N = 32;
  constexpr int CTA_STRIDE = 1;
  constexpr int K_STAGE = 3;

  TORCH_CHECK(M % CTA_M == 0, "M must be divisible by CTA_M");
  TORCH_CHECK(N % CTA_N == 0, "N must be divisible by CTA_N");
  TORCH_CHECK(K % CTA_K == 0, "K must be divisible by CTA_K");

  size_t smem_max = std::max(
      (CTA_M * CTA_K + CTA_N * CTA_K) * sizeof(int8_t) * K_STAGE,
      CTA_M * CTA_N * sizeof(__nv_bfloat16)
  );


  auto kernel_func = GemmInt8SharedRegPipelineBF16<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, CTA_STRIDE, K_STAGE, true, true,ScaleMulMode::kMode1>;

  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(CTA_STRIDE, M / CTA_M, (N + CTA_N * CTA_STRIDE - 1) / (CTA_N * CTA_STRIDE));
  dim3 block(32, (CTA_M / WARP_M) * (CTA_N / WARP_N));

  kernel_func<<<grid, block, smem_max, stream>>>(
      input.data_ptr<int8_t>(),
      weight.data_ptr<int8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scale_input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scale_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(sum_input.data_ptr()),
      zp_weight.data_ptr<int16_t>(),
      M, N, K
  );

  return output;
}


torch::Tensor w8a8_of16_bias_weight_sym(torch::Tensor input,
                      torch::Tensor weight,
                      torch::Tensor bias,
                      torch::Tensor scale_input,
                      torch::Tensor scale_weight)
{
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  CHECK_CUDA(scale_input);
  CHECK_CUDA(scale_weight);

  c10::cuda::OptionalCUDAGuard guard(input.device());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);
  CHECK_CONTIGUOUS(scale_input);
  CHECK_CONTIGUOUS(scale_weight);

  CHECK_DTYPE(input, torch::kInt8);
  CHECK_DTYPE(weight, torch::kInt8);
  CHECK_DTYPE(bias, torch::kHalf);
  CHECK_DTYPE(scale_input, torch::kHalf);
  CHECK_DTYPE(scale_weight, torch::kHalf);

  const int M = input.size(0);
  const int N = weight.size(0);
  const int K = input.size(1);

  CHECK_SHAPE(input, M, K);
  CHECK_SHAPE(weight, N, K);
  CHECK_SHAPE(bias, N);
  CHECK_SHAPE(scale_input, M);
  CHECK_SHAPE(scale_weight, N);

  
  at::Tensor output = torch::empty({input.size(0), weight.size(0)}, input.options().dtype(torch::kHalf));
  // printf("%d,%d,%d,%d,%d",input.device().index(),weight.device().index(),scale_input.device().index(),bias.device().index(),output.device().index());

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();


  const int CTA_M = 128;
  const int CTA_N = 128;
  const int CTA_K = 64;
  constexpr int WARP_M = 128;
  constexpr int WARP_N = 32;
  constexpr int CTA_STRIDE = 1;
  constexpr int K_STAGE = 3;

  assert(M % CTA_M == 0);
  assert(N % CTA_N == 0);
  assert(K % CTA_K == 0);

  size_t smem_max = std::max((CTA_M * CTA_K + CTA_N * CTA_K) * sizeof(int8_t) * K_STAGE, CTA_M * CTA_N * sizeof(half));

  auto kernel_func = GemmInt8SharedRegPipelineV2<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, CTA_STRIDE, OutputDtype::kFloat16, K_STAGE, true, false>;

  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(CTA_STRIDE, M / CTA_M, div_ceil(N / CTA_N, CTA_STRIDE));
  dim3 block(32, (CTA_M / WARP_M) * (CTA_N / WARP_N));

  kernel_func<<<grid, block, smem_max, stream>>>(
    input.data_ptr<int8_t>(),
    weight.data_ptr<int8_t>(),
    reinterpret_cast<half*>(output.data_ptr()),
    nullptr,
    reinterpret_cast<half*>(bias.data_ptr()),
    reinterpret_cast<half*>(scale_input.data_ptr()),
    reinterpret_cast<half*>(scale_weight.data_ptr()),
    nullptr,
    nullptr,
    M, N, K);

  return output;
}

torch::Tensor w8a8_bf16_bias_weight_sym(torch::Tensor input,
                      torch::Tensor weight,
                      torch::Tensor bias,
                      torch::Tensor scale_input,
                      torch::Tensor scale_weight)
{
  CHECK_CUDA(input);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  CHECK_CUDA(scale_input);
  CHECK_CUDA(scale_weight);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);
  CHECK_CONTIGUOUS(scale_input);
  CHECK_CONTIGUOUS(scale_weight);

  CHECK_DTYPE(input, torch::kInt8);
  CHECK_DTYPE(weight, torch::kInt8);
  CHECK_DTYPE(bias, torch::kBFloat16);
  CHECK_DTYPE(scale_input, torch::kBFloat16);
  CHECK_DTYPE(scale_weight, torch::kBFloat16);

  const int M = input.size(0);
  const int N = weight.size(0);
  const int K = input.size(1);

  CHECK_SHAPE(input, M, K);
  CHECK_SHAPE(weight, N, K);
  CHECK_SHAPE(bias, N);
  CHECK_SHAPE(scale_input, M);
  CHECK_SHAPE(scale_weight, N);

  at::Tensor output = torch::empty({input.size(0), weight.size(0)}, input.options().dtype(torch::kBFloat16));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int CTA_M = 128;
  const int CTA_N = 128;
  const int CTA_K = 64;
  constexpr int WARP_M = 128;
  constexpr int WARP_N = 32;
  constexpr int CTA_STRIDE = 1;
  constexpr int K_STAGE = 3;

  assert(M % CTA_M == 0);
  assert(N % CTA_N == 0);
  assert(K % CTA_K == 0);

  size_t smem_max = std::max((CTA_M * CTA_K + CTA_N * CTA_K) * sizeof(int8_t) * K_STAGE, CTA_M * CTA_N * sizeof(__nv_bfloat16));

  auto kernel_func = GemmInt8SharedRegPipelineBF16<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, CTA_STRIDE, K_STAGE, true, false,ScaleMulMode::kMode1>;

  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(CTA_STRIDE, M / CTA_M, div_ceil(N / CTA_N, CTA_STRIDE));
  dim3 block(32, (CTA_M / WARP_M) * (CTA_N / WARP_N));

  kernel_func<<<grid, block, smem_max, stream>>>(
    input.data_ptr<int8_t>(),
    weight.data_ptr<int8_t>(),
    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(scale_input.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(scale_weight.data_ptr()),
    nullptr,
    nullptr,
    M, N, K);

  return output;
}


torch::Tensor w8a8_o32(torch::Tensor input,
                      torch::Tensor weight)
{
  CHECK_CUDA(input);
  CHECK_CUDA(weight);

  c10::cuda::OptionalCUDAGuard guard(input.device().index());  // checkout device to input-tensor device

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);


  CHECK_DTYPE(input, torch::kInt8);
  CHECK_DTYPE(weight, torch::kInt8);

  const int M = input.size(0);
  const int N = weight.size(0);
  const int K = input.size(1);

  CHECK_SHAPE(input, M, K);
  CHECK_SHAPE(weight, N, K);

  at::Tensor output = torch::empty({input.size(0), weight.size(0)}, input.options().dtype(torch::kInt32));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int CTA_M = 128;
  const int CTA_N = 128;
  const int CTA_K = 64;
  constexpr int WARP_M = 128;
  constexpr int WARP_N = 32;
  constexpr int CTA_STRIDE = 1;
  constexpr int K_STAGE = 3;

  assert(M % CTA_M == 0);
  assert(N % CTA_N == 0);
  assert(K % CTA_K == 0);

  size_t smem_max = std::max((CTA_M * CTA_K + CTA_N * CTA_K) * sizeof(int8_t) * K_STAGE, CTA_M * CTA_N * sizeof(int32_t));

  // std::cout<<"smem_max: "<<smem_max / 1024<<"kB"<<std::endl;

  auto kernel_func = GemmInt8SharedRegPipelineV2<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, CTA_STRIDE, OutputDtype::kInt32, K_STAGE, false, false>;

  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(CTA_STRIDE, M / CTA_M, div_ceil(N / CTA_N, CTA_STRIDE));
  dim3 block(32, (CTA_M / WARP_M) * (CTA_N / WARP_N));

  kernel_func<<<grid, block, smem_max, stream>>>(
    input.data_ptr<int8_t>(),
    weight.data_ptr<int8_t>(),
    nullptr,
    output.data_ptr<int32_t>(),
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    M, N, K);

  return output;
}