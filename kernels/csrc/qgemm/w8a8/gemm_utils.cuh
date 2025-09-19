#pragma once
#include "../../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../cp_async.cuh"
#include "../../mma.cuh"
#include "../../permuted_smem.cuh"

#define MMA_M 16
#define MMA_N 16
#define MMA_K 32
#define WARP_SIZE 32
#define PACK_SIZE 16 // how many elements a load/store 128b instruction can manage for A/B

enum class ScaleMulMode {
  kMode1,
  kMode2, // number aligned with Qserve
};

enum class OutputDtype {
  kInt32,
  kFloat16,
};

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ uint32_t get_warp_id()
{
  return threadIdx.y;
}

__device__ __forceinline__ uint32_t get_lane_id()
{
  return threadIdx.x;
}

template <uint32_t num_warps_m, uint32_t num_warps_n>
__device__ __forceinline__ uint32_t get_warp_idx_m()
{
  return get_warp_id() / num_warps_n;
}

template <uint32_t num_warps_m, uint32_t num_warps_n>
__device__ __forceinline__ uint32_t get_warp_idx_n()
{
  return get_warp_id() % num_warps_n;
}

template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, 
          uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void load_AB_global_smem(const int8_t *ptr_base, // ptr_base is aligned to warp_id
                                                    uint32_t K,
                                                    smem_t<swizzle_mode, stride> smem)
{
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
  
  const uint32_t warp_id = get_warp_id();
  const uint32_t lane_id = get_lane_id();

  const int8_t *lane_ptr = ptr_base + lane_id / global_to_shared_line_lanes * K + (lane_id % global_to_shared_line_lanes) * PACK_SIZE;
  uint32_t offset = smem.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_per_iter * smem_iters_col + lane_id / global_to_shared_line_lanes, lane_id % global_to_shared_line_lanes);

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++)
    {
      smem.load_128b_async(offset, lane_ptr);
      lane_ptr += global_to_shared_line_lanes * PACK_SIZE;
      offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(offset);
    }

    offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(offset - (smem_iters_row * global_to_shared_line_lanes));
    lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * K) - (smem_iters_row * global_to_shared_line_lanes * PACK_SIZE));
  }
}

template <uint32_t num_warps_m, uint32_t num_warps_n, uint32_t num_tiles_m, uint32_t num_tiles_n, SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void share_to_reg_A(smem_t<swizzle_mode, stride> current_smem_A, uint32_t RA[][4], uint32_t padding)
{
  const uint32_t warp_id = get_warp_id();
  const uint32_t lane_id = get_lane_id();

  uint32_t offset_A = current_smem_A.get_permuted_offset(get_warp_idx_m<num_warps_m, num_warps_n>() * (num_tiles_m * MMA_M) + lane_id % 16, lane_id / 16 + padding);

#pragma unroll
  for (uint32_t i = 0; i < num_tiles_m; i++)
  {
    current_smem_A.ldmatrix_m8n8x4(offset_A, RA[i]);
    offset_A = current_smem_A.advance_offset_by_row<MMA_M>(offset_A);
  }
}

template <uint32_t num_warps_m, uint32_t num_warps_n, uint32_t num_tiles_m, uint32_t num_tiles_n, SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void share_to_reg_B(smem_t<swizzle_mode, stride> current_smem_B, uint32_t RB[][4], uint32_t padding)
{
  const uint32_t warp_id = get_warp_id();
  const uint32_t lane_id = get_lane_id();

  uint32_t offset_B = current_smem_B.get_permuted_offset(get_warp_idx_n<num_warps_m, num_warps_n>() * (num_tiles_n * MMA_N) + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2 + padding);
  
#pragma unroll
  for (uint32_t i = 0; i < num_tiles_n; i++)
  {
    current_smem_B.ldmatrix_m8n8x4(offset_B, RB[i]);
    offset_B = current_smem_B.advance_offset_by_row<MMA_N>(offset_B);
  }
}

template <uint32_t num_tiles_m, uint32_t num_tiles_n>
__device__ __forceinline__ void tensor_core_mma(int32_t RC[][num_tiles_n][8], uint32_t RA[][4], uint32_t RB[][4])
{
#pragma unroll
  for (uint32_t i = 0; i < num_tiles_m; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < num_tiles_n; j++)
    {
      uint32_t j_s = (i % 2) ? (num_tiles_n - j - 1) : j;
      mma::mma_sync_m16n16k32_row_col_s8s8s32(RC[i][j_s], RA[i], RB[j_s]);
    }
  }
}