// Matrix products and lookups on quantized weights. A matrix crosses the memory bus as one byte
// (INT8) or half a byte (INT4) per weight; the kernels widen the integers to FP32 as they read
// them and multiply with FP16 activations, summing in FP32.

#include <bit>

#include "common.cuh"
#include "kvbound/error.h"

namespace kvb::kernels {
namespace {

// Weights one 16-byte load holds.
constexpr int kInt8PerLoad = 16;
constexpr int kInt4PerLoad = 32;
// Output channels per block: one warp each.
constexpr int kChannelsPerBlock = kBlockWarps;

// The 32 four-bit values of one load, in memory order: the low half of each byte comes first.
__device__ inline void unpack_int4(uint4 load, float* out) {
  const unsigned words[4] = {load.x, load.y, load.z, load.w};
  for (int w = 0; w < 4; ++w) {
    for (int n = 0; n < 8; ++n) {
      out[8 * w + n] = static_cast<float>(static_cast<int>(words[w] << (28 - 4 * n)) >> 28);
    }
  }
}

// Dot product of N weights with N consecutive FP16 inputs, read 16 bytes at a time.
template <int N>
__device__ inline float dot_f16(const float* w, const uint4* x) {
  float sum = 0.0f;
  for (int v = 0; v < N / 8; ++v) {
    const uint4 h = x[v];
    const float* u = w + 8 * v;
    sum += u[0] * f16_bits_to_float(h.x) + u[1] * f16_bits_to_float(h.x >> 16) +
           u[2] * f16_bits_to_float(h.y) + u[3] * f16_bits_to_float(h.y >> 16) +
           u[4] * f16_bits_to_float(h.z) + u[5] * f16_bits_to_float(h.z >> 16) +
           u[6] * f16_bits_to_float(h.w) + u[7] * f16_bits_to_float(h.w >> 16);
  }
  return sum;
}

__device__ inline unsigned f16_pair(float lo, float hi) {
  return static_cast<unsigned>(__half_as_ushort(__float2half(lo))) |
         static_cast<unsigned>(__half_as_ushort(__float2half(hi))) << 16;
}

// One warp per output channel. Lane l reads loads l, l + 32, ... of the channel's row, so each
// trip of the loop has the warp read one contiguous 512-byte stretch, and multiplies every load
// with the matching inputs of each row of x before issuing the next. kRows is the most rows of
// x one launch takes: a compile-time bound keeps the per-row sums in registers.
// Weights are read with a streaming hint: each is used once, and keeping them out of L1 leaves
// it to the rows of x, which every warp of the block reads.
//
// With kInt4, `scale` holds one FP16 per group of 1 << group_loads_shift loads of each row;
// otherwise one per row.
template <bool kInt4, int kRows, bool kF32Out>
__global__ void linear_quantized_kernel(const uint4* weight, const __half* scale, const __half* x,
                                        void* y, int n_rows, int n_in, int n_out,
                                        int group_loads_shift) {
  constexpr int kPerLoad = kInt4 ? kInt4PerLoad : kInt8PerLoad;
  const int lane = threadIdx.x % kWarpSize;
  const std::int64_t channel =
      static_cast<std::int64_t>(blockIdx.x) * kChannelsPerBlock + threadIdx.x / kWarpSize;
  if (channel >= n_out) return;
  const int n_loads = n_in / kPerLoad;
  const uint4* row = weight + channel * n_loads;
  const __half* row_scale = scale + (kInt4 ? channel * (n_loads >> group_loads_shift) : channel);

  float sums[kRows];
  for (int r = 0; r < kRows; ++r) sums[r] = 0.0f;
  for (int load = lane; load < n_loads; load += kWarpSize) {
    float w[kPerLoad];
    float load_scale = 1.0f;
    if constexpr (kInt4) {
      unpack_int4(__ldcs(row + load), w);
      load_scale = __half2float(row_scale[load >> group_loads_shift]);
    } else {
      unpack_int8(__ldcs(row + load), w);
    }
    for (int r = 0; r < kRows; ++r) {
      if (r >= n_rows) break;
      const uint4* inputs = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(r) * n_in);
      sums[r] += load_scale * dot_f16<kPerLoad>(w, inputs + load * (kPerLoad / 8));
    }
  }
  const float channel_scale = kInt4 ? 1.0f : __half2float(row_scale[0]);
  for (int r = 0; r < kRows; ++r) {
    if (r >= n_rows) break;
    const float sum = warp_sum(sums[r]) * channel_scale;
    if (lane != 0) continue;
    const std::size_t at = static_cast<std::size_t>(r) * n_out + channel;
    if constexpr (kF32Out) {
      static_cast<float*>(y)[at] = sum;
    } else {
      static_cast<__half*>(y)[at] = __float2half(sum);
    }
  }
}

// One thread per 16-byte load of the channels given: writes its weights as FP16. `weight` and
// `scale` start at the first of those channels.
template <bool kInt4>
__global__ void dequantize_kernel(const uint4* weight, const __half* scale, uint4* out, int n_loads,
                                  std::size_t n, int group_loads_shift) {
  constexpr int kPerLoad = kInt4 ? kInt4PerLoad : kInt8PerLoad;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const std::size_t channel = i / n_loads;
  const int load = static_cast<int>(i % n_loads);
  float w[kPerLoad];
  float s;
  if constexpr (kInt4) {
    unpack_int4(weight[i], w);
    s = __half2float(scale[channel * (n_loads >> group_loads_shift) + (load >> group_loads_shift)]);
  } else {
    unpack_int8(weight[i], w);
    s = __half2float(scale[channel]);
  }
  for (int v = 0; v < kPerLoad / 8; ++v) {
    const float* u = w + 8 * v;
    out[i * (kPerLoad / 8) + v] =
        make_uint4(f16_pair(u[0] * s, u[1] * s), f16_pair(u[2] * s, u[3] * s),
                   f16_pair(u[4] * s, u[5] * s), f16_pair(u[6] * s, u[7] * s));
  }
}

// One thread per output element, like the FP16 lookup.
template <bool kInt4>
__global__ void embedding_quantized_kernel(const std::uint8_t* table, const __half* scale,
                                           const StepRow* rows, __half* out, int dim, std::size_t n,
                                           int group_shift) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int d = static_cast<int>(i % dim);
  const std::size_t token = static_cast<std::size_t>(rows[i / dim].token_id);
  if constexpr (kInt4) {
    const unsigned byte = table[token * (dim / 2) + d / 2];
    const int q = static_cast<int>((byte >> (4 * (d & 1))) << 28) >> 28;
    const float s = __half2float(scale[token * (dim >> group_shift) + (d >> group_shift)]);
    out[i] = __float2half(static_cast<float>(q) * s);
  } else {
    const int q = static_cast<std::int8_t>(table[token * dim + d]);
    out[i] = __float2half(static_cast<float>(q) * __half2float(scale[token]));
  }
}

// log2 of the 16-byte loads one scale covers.
int group_loads_shift(const WeightRef& w) {
  if (w.scheme != WeightScheme::Int4Grouped) return 0;
  return std::countr_zero(static_cast<unsigned>(w.group_size / kInt4PerLoad));
}

void check_quantized(const WeightRef& w, int n_in) {
  if (w.scheme == WeightScheme::F16) throw Error("not a quantized matrix");
  check_row_length({.scheme = w.scheme, .group_size = w.group_size}, n_in);
}

template <bool kInt4, bool kF32Out>
void launch_linear(const WeightRef& w, const f16* x, void* y, int n_rows, int n_in, int n_out,
                   cudaStream_t stream) {
  const unsigned grid = static_cast<unsigned>((n_out + kChannelsPerBlock - 1) / kChannelsPerBlock);
  const uint4* weight = static_cast<const uint4*>(w.data);
  const __half* scale = as_half(w.scale);
  const int shift = group_loads_shift(w);
#define KVB_LAUNCH(rows)                                                             \
  linear_quantized_kernel<kInt4, rows, kF32Out><<<grid, kBlockThreads, 0, stream>>>( \
      weight, scale, as_half(x), y, n_rows, n_in, n_out, shift)
  if (n_rows == 1) {
    KVB_LAUNCH(1);
  } else if (n_rows == 2) {
    KVB_LAUNCH(2);
  } else if (n_rows <= 4) {
    KVB_LAUNCH(4);
  } else if (n_rows <= 8) {
    KVB_LAUNCH(8);
  } else {
    KVB_LAUNCH(16);
  }
#undef KVB_LAUNCH
  KVB_CUDA(cudaGetLastError());
}

}  // namespace

void linear_quantized(const WeightRef& w, const f16* x, void* y, int n_rows, int n_in, int n_out,
                      bool f32_out, cudaStream_t stream) {
  check_quantized(w, n_in);
  const bool int4 = w.scheme == WeightScheme::Int4Grouped;
  const std::size_t out_bytes = f32_out ? sizeof(float) : sizeof(f16);
  // The matrix is read once for every kQuantizedLinearMaxRows rows.
  for (int first = 0; first < n_rows; first += kQuantizedLinearMaxRows) {
    const int rows = std::min(kQuantizedLinearMaxRows, n_rows - first);
    const f16* xs = x + static_cast<std::size_t>(first) * n_in;
    void* ys = static_cast<std::byte*>(y) + static_cast<std::size_t>(first) * n_out * out_bytes;
    if (int4 && f32_out) launch_linear<true, true>(w, xs, ys, rows, n_in, n_out, stream);
    if (int4 && !f32_out) launch_linear<true, false>(w, xs, ys, rows, n_in, n_out, stream);
    if (!int4 && f32_out) launch_linear<false, true>(w, xs, ys, rows, n_in, n_out, stream);
    if (!int4 && !f32_out) launch_linear<false, false>(w, xs, ys, rows, n_in, n_out, stream);
  }
}

void dequantize(const WeightRef& w, f16* out, int n_in, int first_channel, int n_channels,
                cudaStream_t stream) {
  check_quantized(w, n_in);
  const bool int4 = w.scheme == WeightScheme::Int4Grouped;
  const int n_loads = n_in / (int4 ? kInt4PerLoad : kInt8PerLoad);
  const std::size_t n = static_cast<std::size_t>(n_channels) * n_loads;
  const std::size_t first = static_cast<std::size_t>(first_channel);
  const uint4* weight = static_cast<const uint4*>(w.data) + first * n_loads;
  uint4* dst = reinterpret_cast<uint4*>(out);
  if (int4) {
    const int shift = group_loads_shift(w);
    const __half* scale = as_half(w.scale) + first * (n_loads >> shift);
    dequantize_kernel<true>
        <<<blocks_for(n), kBlockThreads, 0, stream>>>(weight, scale, dst, n_loads, n, shift);
  } else {
    dequantize_kernel<false><<<blocks_for(n), kBlockThreads, 0, stream>>>(
        weight, as_half(w.scale) + first, dst, n_loads, n, 0);
  }
  KVB_CUDA(cudaGetLastError());
}

void embedding_quantized(const WeightRef& table, const StepRow* rows, f16* out, int n_tokens,
                         int dim, cudaStream_t stream) {
  check_quantized(table, dim);
  const std::size_t n = static_cast<std::size_t>(n_tokens) * dim;
  const std::uint8_t* data = static_cast<const std::uint8_t*>(table.data);
  if (table.scheme == WeightScheme::Int4Grouped) {
    embedding_quantized_kernel<true><<<blocks_for(n), kBlockThreads, 0, stream>>>(
        data, as_half(table.scale), rows, as_half(out), dim, n,
        std::countr_zero(static_cast<unsigned>(table.group_size)));
  } else {
    embedding_quantized_kernel<false><<<blocks_for(n), kBlockThreads, 0, stream>>>(
        data, as_half(table.scale), rows, as_half(out), dim, n, 0);
  }
  KVB_CUDA(cudaGetLastError());
}

}  // namespace kvb::kernels
