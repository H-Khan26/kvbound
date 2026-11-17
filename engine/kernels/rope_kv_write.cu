// The step from the stacked q|k|v product to attention in one kernel: rotate q and k by each
// row's position, store q for attention, and store k and v in the KV cache. Unfused, the same
// work is two RoPE launches and a cache write, each reading back what the one before wrote.

#include "common.cuh"
#include "kvbound/error.h"

namespace kvb::kernels {
namespace {

constexpr int kMaxHeadDim = 256;

__device__ inline float round_f16(float x) { return __half2float(__float2half(x)); }

// One warp per head vector of a token: n_heads q vectors, then n_kv_heads k and n_kv_heads v
// vectors, the order the stacked product writes them in. The warp stages the vector in shared
// memory as FP16-rounded floats, so every result is what the unfused kernels would store, then
// writes it out. An INT8 cache gets the vector's largest magnitude over 127 as its FP16 scale,
// as in kv_write.
template <bool kInt8>
__global__ void rope_kv_write_kernel(const __half* qkv, const StepRow* rows, const float2* cos_sin,
                                     __half* q_out, KvView kv, int n_heads, int n_kv_heads,
                                     int head_dim, std::size_t n_vectors) {
  __shared__ float staged[kBlockWarps][kMaxHeadDim];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const std::size_t vec = static_cast<std::size_t>(blockIdx.x) * kBlockWarps + warp;
  if (vec >= n_vectors) return;
  const int per_token = n_heads + 2 * n_kv_heads;
  const std::size_t token = vec / per_token;
  const int head = static_cast<int>(vec % per_token);
  const __half* src = qkv + vec * head_dim;
  const StepRow row = rows[token];
  float* v = staged[warp];

  const int half_dim = head_dim / 2;
  if (head < n_heads + n_kv_heads) {
    // HF Llama pairs dimension j with j + head_dim/2.
    const float2* angles = cos_sin + static_cast<std::size_t>(row.position) * half_dim;
    for (int j = lane; j < half_dim; j += kWarpSize) {
      const float2 cs = angles[j];
      const float a = __half2float(src[j]);
      const float b = __half2float(src[j + half_dim]);
      v[j] = round_f16(a * cs.x - b * cs.y);
      v[j + half_dim] = round_f16(b * cs.x + a * cs.y);
    }
  } else {
    for (int d = lane; d < head_dim; d += kWarpSize) v[d] = __half2float(src[d]);
  }
  __syncwarp();

  if (head < n_heads) {
    __half* dst = q_out + (token * n_heads + head) * head_dim;
    for (int d = lane; d < head_dim; d += kWarpSize) dst[d] = __float2half(v[d]);
    return;
  }
  const bool is_v = head >= n_heads + n_kv_heads;
  const int kv_head = (head - n_heads) % n_kv_heads;
  const unsigned slot = static_cast<unsigned>(row.kv_slot);
  const std::int64_t block = slot / static_cast<unsigned>(kv.block_size);
  const std::int64_t in_block = slot % static_cast<unsigned>(kv.block_size);
  const std::int64_t dst =
      block * kv.block_stride + kv_head * kv.head_stride + in_block * kv.pos_stride;
  if constexpr (kInt8) {
    float largest = 0.0f;
    for (int d = lane; d < head_dim; d += kWarpSize) largest = fmaxf(largest, fabsf(v[d]));
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
    }
    const __half stored = __float2half(largest / 127.0f);
    const float scale = __half2float(stored);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    std::int8_t* out = static_cast<std::int8_t*>(is_v ? kv.v : kv.k) + dst;
    for (int d = lane; d < head_dim; d += kWarpSize) {
      const float q = rintf(v[d] * inverse);
      out[d * kv.dim_stride] = static_cast<std::int8_t>(fminf(fmaxf(q, -127.0f), 127.0f));
    }
    if (lane == 0) {
      const std::int64_t scale_at =
          block * kv.scale_block_stride + kv_head * kv.scale_head_stride + in_block;
      static_cast<__half*>(is_v ? kv.v_scale : kv.k_scale)[scale_at] = stored;
    }
  } else {
    __half* out = static_cast<__half*>(is_v ? kv.v : kv.k) + dst;
    for (int d = lane; d < head_dim; d += kWarpSize) out[d * kv.dim_stride] = __float2half(v[d]);
  }
}

}  // namespace

void rope_kv_write(const f16* qkv, const StepRow* rows, const float* cos_sin, f16* q,
                   const KvView& kv, int n_tokens, int n_heads, int n_kv_heads, int head_dim,
                   cudaStream_t stream) {
  if (head_dim > kMaxHeadDim || head_dim % 2 != 0) {
    throw Error("rope_kv_write supports even head_dim up to 256");
  }
  const std::size_t n_vectors = static_cast<std::size_t>(n_tokens) * (n_heads + 2 * n_kv_heads);
  const unsigned grid = static_cast<unsigned>((n_vectors + kBlockWarps - 1) / kBlockWarps);
  const auto* angles = reinterpret_cast<const float2*>(cos_sin);
  if (kv.dtype == DType::I8) {
    rope_kv_write_kernel<true><<<grid, kBlockThreads, 0, stream>>>(
        as_half(qkv), rows, angles, as_half(q), kv, n_heads, n_kv_heads, head_dim, n_vectors);
  } else {
    rope_kv_write_kernel<false><<<grid, kBlockThreads, 0, stream>>>(
        as_half(qkv), rows, angles, as_half(q), kv, n_heads, n_kv_heads, head_dim, n_vectors);
  }
  KVB_CUDA(cudaGetLastError());
}

}  // namespace kvb::kernels
