// Generic SRAM/shared-memory attention CUDA implementations.
//
// Narrow Phase-2 scope:
// - pre-shaped Q/K/V only (`[B,H,S,D]`)
// - CUDA only
// - fp16 only
// - `head_dim=128`
// - shared-memory staged KV tiles + online softmax accumulation, with optional internal split-KV reduction dispatch

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int64_t kAttentionSramHeadDim = 128;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kKvTileTokens = 32;
constexpr int kSplitKvMaxSplits = 8;
constexpr int64_t kSplitKvTempBudgetBytes = 256LL * 1024LL * 1024LL;
constexpr int kValuesPerLane = kAttentionSramHeadDim / kWarpSize;
static_assert(kValuesPerLane == 4, "Expected head_dim=128 to map to 4 values per warp lane.");

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
}

void check_same_device(const torch::Tensor& reference, const torch::Tensor& value, const char* name) {
  TORCH_CHECK(
      reference.device() == value.device(),
      name,
      " must be on the same device as q (q=",
      reference.device(),
      " got=",
      value.device(),
      ").");
}

int64_t checked_mul_int64(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0, label, " must use non-negative factors.");
  if (left == 0 || right == 0) {
    return 0;
  }
  TORCH_CHECK(left <= std::numeric_limits<int64_t>::max() / right, label, " overflow.");
  return left * right;
}

__host__ __device__ __forceinline__ int64_t ceil_div_int64(int64_t value, int64_t divisor) {
  if (value == 0) {
    return 0;
  }
  return 1 + ((value - 1) / divisor);
}

bool safe_mul_int64(int64_t left, int64_t right, int64_t* out) {
  if (left < 0 || right < 0) {
    return false;
  }
  if (left == 0 || right == 0) {
    *out = 0;
    return true;
  }
  if (left > std::numeric_limits<int64_t>::max() / right) {
    return false;
  }
  *out = left * right;
  return true;
}

bool safe_add_int64(int64_t left, int64_t right, int64_t* out) {
  if (left < 0 || right < 0) {
    return false;
  }
  if (left > std::numeric_limits<int64_t>::max() - right) {
    return false;
  }
  *out = left + right;
  return true;
}

bool is_split_eligible(int64_t num_k_tiles, int64_t num_splits) {
  if (num_splits <= 1) {
    return true;
  }
  return ceil_div_int64(num_k_tiles, num_splits) != ceil_div_int64(num_k_tiles, num_splits - 1);
}

int choose_splitkv_num_splits(int64_t batch, int64_t heads, int64_t q_len, int64_t kv_len, int64_t num_sms) {
  if (num_sms <= 0) {
    return 1;
  }
  const int64_t num_k_tiles = ceil_div_int64(kv_len, kKvTileTokens);
  if (num_k_tiles <= 1) {
    return 1;
  }
  const int64_t num_q_blocks = ceil_div_int64(q_len, kWarpsPerBlock);
  const int64_t work_tiles = checked_mul_int64(
      checked_mul_int64(batch, heads, "split heuristic work tiles"),
      num_q_blocks,
      "split heuristic work tiles");
  const int64_t occupancy_target = std::max<int64_t>(1, ceil_div_int64(num_sms * 4, 5));
  const int64_t requested_splits = ceil_div_int64(occupancy_target, std::max<int64_t>(int64_t{1}, work_tiles));
  int64_t num_splits = std::min<int64_t>(
      std::min<int64_t>(requested_splits, num_k_tiles),
      static_cast<int64_t>(kSplitKvMaxSplits));
  if (num_splits <= 1) {
    return 1;
  }
  while (num_splits > 1 && !is_split_eligible(num_k_tiles, num_splits)) {
    --num_splits;
  }
  return static_cast<int>(std::max<int64_t>(1, num_splits));
}

bool try_compute_splitkv_temp_bytes(int64_t num_splits, int64_t batch, int64_t heads, int64_t q_len, int64_t* out_bytes) {
  int64_t stats_elements = 0;
  int64_t acc_elements = 0;
  int64_t total_elements = 0;
  int64_t stats_twice = 0;
  if (!safe_mul_int64(num_splits, batch, &stats_elements)) {
    return false;
  }
  if (!safe_mul_int64(stats_elements, heads, &stats_elements)) {
    return false;
  }
  if (!safe_mul_int64(stats_elements, q_len, &stats_elements)) {
    return false;
  }
  if (!safe_mul_int64(stats_elements, kAttentionSramHeadDim, &acc_elements)) {
    return false;
  }
  if (!safe_add_int64(stats_elements, stats_elements, &stats_twice)) {
    return false;
  }
  if (!safe_add_int64(stats_twice, acc_elements, &total_elements)) {
    return false;
  }
  return safe_mul_int64(total_elements, static_cast<int64_t>(sizeof(float)), out_bytes);
}

__host__ __device__ __forceinline__ int64_t split_stat_index(
    int64_t split_index,
    int64_t batch_index,
    int64_t head_index,
    int64_t q_index,
    int64_t batch,
    int64_t heads,
    int64_t q_len) {
  return (((split_index * batch) + batch_index) * heads + head_index) * q_len + q_index;
}

__host__ __device__ __forceinline__ int64_t split_acc_index(
    int64_t split_index,
    int64_t batch_index,
    int64_t head_index,
    int64_t q_index,
    int64_t dim_index,
    int64_t batch,
    int64_t heads,
    int64_t q_len) {
  return ((((split_index * batch) + batch_index) * heads + head_index) * q_len + q_index) *
          kAttentionSramHeadDim +
      dim_index;
}

template <typename scalar_t>
__global__ void rope_blhd_inplace_kernel(
    scalar_t* __restrict__ x,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    int64_t batch,
    int64_t seq_len,
    int64_t heads,
    int64_t head_dim) {
  const int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t half_dim = head_dim / 2;
  const int64_t total_pairs = batch * seq_len * heads * half_dim;
  if (pair_index >= total_pairs) {
    return;
  }

  int64_t idx = pair_index;
  const int64_t d_pair = idx % half_dim;
  idx /= half_dim;
  const int64_t h = idx % heads;
  idx /= heads;
  const int64_t l = idx % seq_len;
  const int64_t b = idx / seq_len;

  const int64_t d_even = d_pair * 2;
  const int64_t d_odd = d_even + 1;
  const int64_t x_base = ((b * seq_len + l) * heads + h) * head_dim;
  const int64_t cos_base = l * head_dim;
  const int64_t sin_base = l * head_dim;

  const float even_value = static_cast<float>(x[x_base + d_even]);
  const float odd_value = static_cast<float>(x[x_base + d_odd]);
  const float cos_even = rope_cos[cos_base + d_even];
  const float sin_odd = rope_sin[sin_base + d_odd];
  const float sin_even = rope_sin[sin_base + d_even];
  const float cos_odd = rope_cos[cos_base + d_odd];

  x[x_base + d_even] = static_cast<scalar_t>(even_value * cos_even - odd_value * sin_odd);
  x[x_base + d_odd] = static_cast<scalar_t>(even_value * sin_even + odd_value * cos_odd);
}

void rope_blhd_inplace_cuda_impl(
    const char* op_name,
    torch::Tensor& x_blhd,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin) {
  TORCH_CHECK(x_blhd.dim() == 4, op_name, " expects x as [B,L,H,D].");
  TORCH_CHECK(rope_cos.dim() == 4 && rope_sin.dim() == 4, op_name, " expects rope tensors as [1,L,1,D].");
  check_cuda_tensor(x_blhd, "x");
  check_cuda_tensor(rope_cos, "rope_cos");
  check_cuda_tensor(rope_sin, "rope_sin");
  check_same_device(x_blhd, rope_cos, "rope_cos");
  check_same_device(x_blhd, rope_sin, "rope_sin");

  TORCH_CHECK(x_blhd.is_contiguous(), op_name, " requires x to be contiguous.");
  TORCH_CHECK(rope_cos.is_contiguous(), op_name, " requires rope_cos to be contiguous.");
  TORCH_CHECK(rope_sin.is_contiguous(), op_name, " requires rope_sin to be contiguous.");
  TORCH_CHECK(rope_cos.scalar_type() == torch::kFloat, op_name, " requires rope_cos to be float32.");
  TORCH_CHECK(rope_sin.scalar_type() == torch::kFloat, op_name, " requires rope_sin to be float32.");

  const int64_t batch = x_blhd.size(0);
  const int64_t seq_len = x_blhd.size(1);
  const int64_t heads = x_blhd.size(2);
  const int64_t head_dim = x_blhd.size(3);
  TORCH_CHECK(head_dim > 0 && (head_dim % 2) == 0, op_name, " requires an even head_dim (got ", head_dim, ").");
  TORCH_CHECK(rope_cos.size(0) == 1 && rope_cos.size(2) == 1, op_name, " requires rope_cos to be [1,L,1,D].");
  TORCH_CHECK(rope_sin.size(0) == 1 && rope_sin.size(2) == 1, op_name, " requires rope_sin to be [1,L,1,D].");
  TORCH_CHECK(rope_cos.size(1) == seq_len && rope_cos.size(3) == head_dim, op_name, " rope_cos shape mismatch.");
  TORCH_CHECK(rope_sin.size(1) == seq_len && rope_sin.size(3) == head_dim, op_name, " rope_sin shape mismatch.");

  const int64_t total_pairs = checked_mul_int64(
      checked_mul_int64(
          checked_mul_int64(batch, seq_len, "batch*seq_len"),
          heads,
          "batch*seq_len*heads"),
      head_dim / 2,
      "total_pairs");
  if (total_pairs == 0) {
    return;
  }

  TORCH_CHECK(
      x_blhd.scalar_type() == torch::kFloat16 || x_blhd.scalar_type() == torch::kBFloat16 ||
          x_blhd.scalar_type() == torch::kFloat,
      op_name,
      " supports x dtype float16|bfloat16|float32 (got ",
      c10::toString(x_blhd.scalar_type()),
      ").");

  const int threads = 256;
  const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
  const auto stream = c10::cuda::getCurrentCUDAStream(x_blhd.get_device());

  const float* cos_ptr = rope_cos.data_ptr<float>();
  const float* sin_ptr = rope_sin.data_ptr<float>();
  if (x_blhd.scalar_type() == torch::kFloat16) {
    rope_blhd_inplace_kernel<c10::Half><<<blocks, threads, 0, stream.stream()>>>(
        x_blhd.data_ptr<c10::Half>(), cos_ptr, sin_ptr, batch, seq_len, heads, head_dim);
  } else if (x_blhd.scalar_type() == torch::kBFloat16) {
    rope_blhd_inplace_kernel<c10::BFloat16><<<blocks, threads, 0, stream.stream()>>>(
        x_blhd.data_ptr<c10::BFloat16>(), cos_ptr, sin_ptr, batch, seq_len, heads, head_dim);
  } else {
    rope_blhd_inplace_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        x_blhd.data_ptr<float>(), cos_ptr, sin_ptr, batch, seq_len, heads, head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <int WarpsPerBlock, int KvTileTokens>
__global__ void attention_sram_fwd_standard_kernel_fp16(
    const c10::Half* __restrict__ q_ptr,
    const c10::Half* __restrict__ k_ptr,
    const c10::Half* __restrict__ v_ptr,
    c10::Half* __restrict__ out_ptr,
    int64_t q_len,
    int64_t kv_len,
    int64_t q_stride_batch,
    int64_t q_stride_head,
    int64_t q_stride_seq,
    int64_t q_stride_dim,
    int64_t k_stride_batch,
    int64_t k_stride_head,
    int64_t k_stride_seq,
    int64_t k_stride_dim,
    int64_t v_stride_batch,
    int64_t v_stride_head,
    int64_t v_stride_seq,
    int64_t v_stride_dim,
    int64_t out_stride_batch,
    int64_t out_stride_head,
    int64_t out_stride_seq,
    int64_t out_stride_dim,
    float scale,
    bool is_causal) {
  __shared__ c10::Half s_k[KvTileTokens * kAttentionSramHeadDim];
  __shared__ c10::Half s_v[KvTileTokens * kAttentionSramHeadDim];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_index = threadIdx.x / kWarpSize;
  const int64_t batch_index = blockIdx.z;
  const int64_t head_index = blockIdx.y;
  const int64_t q_index = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_index;
  const bool warp_active = warp_index < WarpsPerBlock && q_index < q_len;

  float q_values[kValuesPerLane];
  float acc_values[kValuesPerLane];
  #pragma unroll
  for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
    q_values[value_index] = 0.0f;
    acc_values[value_index] = 0.0f;
  }

  float row_max = -INFINITY;
  float row_sum = 0.0f;

  if (warp_active) {
    const int64_t q_row_offset =
        batch_index * q_stride_batch + head_index * q_stride_head + q_index * q_stride_seq;
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      const int dim = lane + value_index * kWarpSize;
      q_values[value_index] = static_cast<float>(q_ptr[q_row_offset + dim * q_stride_dim]);
    }
  }

  const int64_t k_head_offset = batch_index * k_stride_batch + head_index * k_stride_head;
  const int64_t v_head_offset = batch_index * v_stride_batch + head_index * v_stride_head;
  // Bottom-right-aligned rectangular causal mask:
  // keep kv_index <= q_index + (kv_len - q_len), which reduces to the square-case rule when q_len == kv_len.
  const int64_t causal_last_kv_index = q_index + (kv_len - q_len);
  // Keep the outer tile loop block-convergent: rectangular causal cutoffs can differ across the 4 query rows in one block.
  for (int64_t kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += KvTileTokens) {
    const int tile_tokens = static_cast<int>(std::min<int64_t>(KvTileTokens, kv_len - kv_tile_start));
    const int tile_elements = tile_tokens * static_cast<int>(kAttentionSramHeadDim);

    for (int shared_index = threadIdx.x; shared_index < tile_elements; shared_index += blockDim.x) {
      const int token_index = shared_index / static_cast<int>(kAttentionSramHeadDim);
      const int dim_index = shared_index % static_cast<int>(kAttentionSramHeadDim);
      const int64_t k_src_offset =
          k_head_offset + (kv_tile_start + token_index) * k_stride_seq + dim_index * k_stride_dim;
      const int64_t v_src_offset =
          v_head_offset + (kv_tile_start + token_index) * v_stride_seq + dim_index * v_stride_dim;
      s_k[shared_index] = k_ptr[k_src_offset];
      s_v[shared_index] = v_ptr[v_src_offset];
    }
    __syncthreads();

    if (warp_active) {
      for (int token_index = 0; token_index < tile_tokens; ++token_index) {
        const int64_t kv_index = kv_tile_start + token_index;
        if (is_causal && kv_index > causal_last_kv_index) {
          break;
        }

        float dot = 0.0f;
        #pragma unroll
        for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
          const int dim = lane + value_index * kWarpSize;
          dot += q_values[value_index] *
              static_cast<float>(s_k[token_index * kAttentionSramHeadDim + dim]);
        }
        dot = warp_reduce_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * scale;

        const float next_row_max = fmaxf(row_max, score);
        const float alpha = __expf(row_max - next_row_max);
        const float beta = __expf(score - next_row_max);
        const float next_row_sum = row_sum * alpha + beta;

        #pragma unroll
        for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
          const int dim = lane + value_index * kWarpSize;
          const float v_value = static_cast<float>(s_v[token_index * kAttentionSramHeadDim + dim]);
          acc_values[value_index] = acc_values[value_index] * alpha + beta * v_value;
        }

        row_max = next_row_max;
        row_sum = next_row_sum;
      }
    }
    __syncthreads();
  }

  if (!warp_active) {
    return;
  }

  const float inv_row_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
  const int64_t out_row_offset =
      batch_index * out_stride_batch + head_index * out_stride_head + q_index * out_stride_seq;
  #pragma unroll
  for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
    const int dim = lane + value_index * kWarpSize;
    out_ptr[out_row_offset + dim * out_stride_dim] =
        static_cast<c10::Half>(acc_values[value_index] * inv_row_sum);
  }
}

template <int WarpsPerBlock, int KvTileTokens>
__global__ void attention_sram_fwd_splitkv_kernel_fp16(
    const c10::Half* __restrict__ q_ptr,
    const c10::Half* __restrict__ k_ptr,
    const c10::Half* __restrict__ v_ptr,
    float* __restrict__ partial_row_max_ptr,
    float* __restrict__ partial_row_sum_ptr,
    float* __restrict__ partial_acc_ptr,
    int64_t batch,
    int64_t heads,
    int64_t q_len,
    int64_t kv_len,
    int64_t num_splits,
    int64_t q_stride_batch,
    int64_t q_stride_head,
    int64_t q_stride_seq,
    int64_t q_stride_dim,
    int64_t k_stride_batch,
    int64_t k_stride_head,
    int64_t k_stride_seq,
    int64_t k_stride_dim,
    int64_t v_stride_batch,
    int64_t v_stride_head,
    int64_t v_stride_seq,
    int64_t v_stride_dim,
    float scale,
    bool is_causal) {
  __shared__ c10::Half s_k[KvTileTokens * kAttentionSramHeadDim];
  __shared__ c10::Half s_v[KvTileTokens * kAttentionSramHeadDim];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_index = threadIdx.x / kWarpSize;
  const int64_t split_batch_index = blockIdx.z;
  const int64_t split_index = split_batch_index % num_splits;
  const int64_t batch_index = split_batch_index / num_splits;
  const int64_t head_index = blockIdx.y;
  const int64_t q_index = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_index;
  const bool warp_active =
      warp_index < WarpsPerBlock && batch_index < batch && head_index < heads && q_index < q_len;

  float q_values[kValuesPerLane];
  float acc_values[kValuesPerLane];
  #pragma unroll
  for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
    q_values[value_index] = 0.0f;
    acc_values[value_index] = 0.0f;
  }

  float row_max = -INFINITY;
  float row_sum = 0.0f;

  if (warp_active) {
    const int64_t q_row_offset =
        batch_index * q_stride_batch + head_index * q_stride_head + q_index * q_stride_seq;
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      const int dim = lane + value_index * kWarpSize;
      q_values[value_index] = static_cast<float>(q_ptr[q_row_offset + dim * q_stride_dim]);
    }
  }

  const int64_t num_k_tiles = ceil_div_int64(kv_len, KvTileTokens);
  const int64_t tiles_per_split = ceil_div_int64(num_k_tiles, num_splits);
  const int64_t split_tile_start = split_index * tiles_per_split;
  const int64_t split_tile_end = std::min<int64_t>(num_k_tiles, split_tile_start + tiles_per_split);
  const int64_t kv_start = split_tile_start * KvTileTokens;
  const int64_t kv_end = std::min<int64_t>(kv_len, split_tile_end * KvTileTokens);
  const int64_t causal_last_kv_index = q_index + (kv_len - q_len);

  const int64_t k_head_offset = batch_index * k_stride_batch + head_index * k_stride_head;
  const int64_t v_head_offset = batch_index * v_stride_batch + head_index * v_stride_head;
  // Keep the outer tile loop block-convergent: rectangular causal cutoffs can differ across the 4 query rows in one block.
  for (int64_t kv_tile_start = kv_start; kv_tile_start < kv_end; kv_tile_start += KvTileTokens) {
    const int tile_tokens = static_cast<int>(std::min<int64_t>(KvTileTokens, kv_end - kv_tile_start));
    const int tile_elements = tile_tokens * static_cast<int>(kAttentionSramHeadDim);

    for (int shared_index = threadIdx.x; shared_index < tile_elements; shared_index += blockDim.x) {
      const int token_index = shared_index / static_cast<int>(kAttentionSramHeadDim);
      const int dim_index = shared_index % static_cast<int>(kAttentionSramHeadDim);
      const int64_t k_src_offset =
          k_head_offset + (kv_tile_start + token_index) * k_stride_seq + dim_index * k_stride_dim;
      const int64_t v_src_offset =
          v_head_offset + (kv_tile_start + token_index) * v_stride_seq + dim_index * v_stride_dim;
      s_k[shared_index] = k_ptr[k_src_offset];
      s_v[shared_index] = v_ptr[v_src_offset];
    }
    __syncthreads();

    if (warp_active) {
      for (int token_index = 0; token_index < tile_tokens; ++token_index) {
        const int64_t kv_index = kv_tile_start + token_index;
        if (is_causal && kv_index > causal_last_kv_index) {
          break;
        }

        float dot = 0.0f;
        #pragma unroll
        for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
          const int dim = lane + value_index * kWarpSize;
          dot += q_values[value_index] *
              static_cast<float>(s_k[token_index * kAttentionSramHeadDim + dim]);
        }
        dot = warp_reduce_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * scale;

        const float next_row_max = fmaxf(row_max, score);
        const float alpha = __expf(row_max - next_row_max);
        const float beta = __expf(score - next_row_max);
        const float next_row_sum = row_sum * alpha + beta;

        #pragma unroll
        for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
          const int dim = lane + value_index * kWarpSize;
          const float v_value = static_cast<float>(s_v[token_index * kAttentionSramHeadDim + dim]);
          acc_values[value_index] = acc_values[value_index] * alpha + beta * v_value;
        }

        row_max = next_row_max;
        row_sum = next_row_sum;
      }
    }
    __syncthreads();
  }

  if (!warp_active) {
    return;
  }

  const int64_t stat_index = split_stat_index(split_index, batch_index, head_index, q_index, batch, heads, q_len);
  partial_row_max_ptr[stat_index] = row_max;
  partial_row_sum_ptr[stat_index] = row_sum;
  #pragma unroll
  for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
    const int dim = lane + value_index * kWarpSize;
    const int64_t acc_index = split_acc_index(split_index, batch_index, head_index, q_index, dim, batch, heads, q_len);
    partial_acc_ptr[acc_index] = acc_values[value_index];
  }
}

template <int WarpsPerBlock>
__global__ void attention_sram_reduce_splitkv_kernel_fp16(
    const float* __restrict__ partial_row_max_ptr,
    const float* __restrict__ partial_row_sum_ptr,
    const float* __restrict__ partial_acc_ptr,
    c10::Half* __restrict__ out_ptr,
    int64_t batch,
    int64_t heads,
    int64_t q_len,
    int64_t num_splits,
    int64_t out_stride_batch,
    int64_t out_stride_head,
    int64_t out_stride_seq,
    int64_t out_stride_dim) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_index = threadIdx.x / kWarpSize;
  const int64_t batch_index = blockIdx.z;
  const int64_t head_index = blockIdx.y;
  const int64_t q_index = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_index;
  if (warp_index >= WarpsPerBlock || batch_index >= batch || head_index >= heads || q_index >= q_len) {
    return;
  }

  float combined_values[kValuesPerLane];
  #pragma unroll
  for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
    combined_values[value_index] = 0.0f;
  }

  float global_row_max = -INFINITY;
  for (int64_t split_index = 0; split_index < num_splits; ++split_index) {
    const int64_t stat_index = split_stat_index(split_index, batch_index, head_index, q_index, batch, heads, q_len);
    global_row_max = fmaxf(global_row_max, partial_row_max_ptr[stat_index]);
  }

  const int64_t out_row_offset =
      batch_index * out_stride_batch + head_index * out_stride_head + q_index * out_stride_seq;
  if (!isfinite(global_row_max)) {
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      const int dim = lane + value_index * kWarpSize;
      out_ptr[out_row_offset + dim * out_stride_dim] = static_cast<c10::Half>(0.0f);
    }
    return;
  }

  float global_row_sum = 0.0f;
  for (int64_t split_index = 0; split_index < num_splits; ++split_index) {
    const int64_t stat_index = split_stat_index(split_index, batch_index, head_index, q_index, batch, heads, q_len);
    const float split_row_sum = partial_row_sum_ptr[stat_index];
    if (split_row_sum <= 0.0f) {
      continue;
    }
    const float weight = __expf(partial_row_max_ptr[stat_index] - global_row_max);
    global_row_sum += split_row_sum * weight;
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      const int dim = lane + value_index * kWarpSize;
      const int64_t acc_index = split_acc_index(split_index, batch_index, head_index, q_index, dim, batch, heads, q_len);
      combined_values[value_index] += partial_acc_ptr[acc_index] * weight;
    }
  }

  const float inv_global_row_sum = global_row_sum > 0.0f ? 1.0f / global_row_sum : 0.0f;
  #pragma unroll
  for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
    const int dim = lane + value_index * kWarpSize;
    out_ptr[out_row_offset + dim * out_stride_dim] =
        static_cast<c10::Half>(combined_values[value_index] * inv_global_row_sum);
  }
}

void check_supported_attention_layout(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.stride(0) >= 0 && tensor.stride(1) >= 0 && tensor.stride(2) >= 0 && tensor.stride(3) == 1 &&
          (tensor.stride(0) > 0 || tensor.size(0) == 1) && (tensor.stride(1) > 0 || tensor.size(1) == 1) &&
          (tensor.stride(2) > 0 || tensor.size(2) == 1) &&
          tensor.is_non_overlapping_and_dense(),
      "attention_sram_v1.attn_fwd requires ",
      name,
      " to use a non-overlapping dense [B,H,S,D] layout with contiguous head_dim lanes; got stride=(",
      tensor.stride(0),
      ", ",
      tensor.stride(1),
      ", ",
      tensor.stride(2),
      ", ",
      tensor.stride(3),
      ").");
}

void validate_attention_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal) {
  check_cuda_tensor(q, "q");
  check_cuda_tensor(k, "k");
  check_cuda_tensor(v, "v");
  check_same_device(q, k, "k");
  check_same_device(q, v, "v");

  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "attention_sram_v1.attn_fwd expects Q/K/V as rank-4 tensors [B,H,S,D].");
  check_supported_attention_layout(q, "q");
  check_supported_attention_layout(k, "k");
  check_supported_attention_layout(v, "v");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16, "attention_sram_v1.attn_fwd supports q dtype float16 only.");
  TORCH_CHECK(k.scalar_type() == torch::kFloat16, "attention_sram_v1.attn_fwd supports k dtype float16 only.");
  TORCH_CHECK(v.scalar_type() == torch::kFloat16, "attention_sram_v1.attn_fwd supports v dtype float16 only.");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "attention_sram_v1.attn_fwd requires matching batch size across Q/K/V.");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "attention_sram_v1.attn_fwd requires matching head count across Q/K/V.");
  TORCH_CHECK(k.size(2) == v.size(2), "attention_sram_v1.attn_fwd requires matching K/V sequence length.");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "attention_sram_v1.attn_fwd requires matching head_dim across Q/K/V.");
  TORCH_CHECK(q.size(3) == kAttentionSramHeadDim, "attention_sram_v1.attn_fwd currently supports head_dim=128 only.");
  TORCH_CHECK(q.size(0) > 0, "attention_sram_v1.attn_fwd requires batch > 0.");
  TORCH_CHECK(q.size(1) > 0, "attention_sram_v1.attn_fwd requires heads > 0.");
  TORCH_CHECK(k.size(2) > 0, "attention_sram_v1.attn_fwd requires K/V sequence length > 0.");
  TORCH_CHECK(!is_causal || q.size(2) <= std::numeric_limits<int32_t>::max(), "attention_sram_v1.attn_fwd causal path requires q sequence length to fit int32.");
}

}  // namespace

torch::Tensor attention_sram_v1_rope_blhd_inplace_cuda(
    torch::Tensor x_blhd,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin) {
  rope_blhd_inplace_cuda_impl("attention_sram_v1.rope_blhd_", x_blhd, rope_cos, rope_sin);
  return x_blhd;
}

torch::Tensor attention_sram_v1_attn_fwd_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal) {
  validate_attention_inputs(q, k, v, is_causal);
  if (q.size(2) == 0) {
    return torch::empty_strided(q.sizes(), q.strides(), q.options());
  }

  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t q_len = q.size(2);
  const int64_t kv_len = k.size(2);

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_strided(q.sizes(), q.strides(), q.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  const auto* device_properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(device_properties != nullptr, "attention_sram_v1.attn_fwd could not read current CUDA device properties.");
  const float scale = 1.0f / std::sqrt(static_cast<float>(kAttentionSramHeadDim));
  int num_splits = choose_splitkv_num_splits(batch, heads, q_len, kv_len, static_cast<int64_t>(device_properties->multiProcessorCount));
  int64_t split_temp_bytes = 0;
  if (num_splits > 1) {
    if (!try_compute_splitkv_temp_bytes(num_splits, batch, heads, q_len, &split_temp_bytes) ||
        split_temp_bytes > kSplitKvTempBudgetBytes) {
      num_splits = 1;
    }
  }

  const dim3 block(kWarpsPerBlock * kWarpSize);
  const unsigned int q_block_count =
      static_cast<unsigned int>(ceil_div_int64(q_len, kWarpsPerBlock));
  const dim3 grid(
      q_block_count,
      static_cast<unsigned int>(heads),
      static_cast<unsigned int>(batch));
  if (num_splits <= 1) {
    attention_sram_fwd_standard_kernel_fp16<kWarpsPerBlock, kKvTileTokens><<<grid, block, 0, stream.stream()>>>(
        q.data_ptr<c10::Half>(),
        k.data_ptr<c10::Half>(),
        v.data_ptr<c10::Half>(),
        out.data_ptr<c10::Half>(),
        q_len,
        kv_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        scale,
        is_causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  auto temp_options = q.options().dtype(torch::kFloat);
  auto partial_row_max = torch::empty({num_splits, batch, heads, q_len}, temp_options);
  auto partial_row_sum = torch::empty({num_splits, batch, heads, q_len}, temp_options);
  auto partial_acc = torch::empty(
      {num_splits, batch, heads, q_len, static_cast<int64_t>(kAttentionSramHeadDim)},
      temp_options);

  const int64_t split_grid_depth = checked_mul_int64(batch, static_cast<int64_t>(num_splits), "attention_sram_v1 split grid depth");
  const dim3 split_grid(
      q_block_count,
      static_cast<unsigned int>(heads),
      static_cast<unsigned int>(split_grid_depth));
  attention_sram_fwd_splitkv_kernel_fp16<kWarpsPerBlock, kKvTileTokens><<<split_grid, block, 0, stream.stream()>>>(
      q.data_ptr<c10::Half>(),
      k.data_ptr<c10::Half>(),
      v.data_ptr<c10::Half>(),
      partial_row_max.data_ptr<float>(),
      partial_row_sum.data_ptr<float>(),
      partial_acc.data_ptr<float>(),
      batch,
      heads,
      q_len,
      kv_len,
      num_splits,
      q.stride(0),
      q.stride(1),
      q.stride(2),
      q.stride(3),
      k.stride(0),
      k.stride(1),
      k.stride(2),
      k.stride(3),
      v.stride(0),
      v.stride(1),
      v.stride(2),
      v.stride(3),
      scale,
      is_causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  attention_sram_reduce_splitkv_kernel_fp16<kWarpsPerBlock><<<grid, block, 0, stream.stream()>>>(
      partial_row_max.data_ptr<float>(),
      partial_row_sum.data_ptr<float>(),
      partial_acc.data_ptr<float>(),
      out.data_ptr<c10::Half>(),
      batch,
      heads,
      q_len,
      num_splits,
      out.stride(0),
      out.stride(1),
      out.stride(2),
      out.stride(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
