// Matrix products on FP16 weights for decode-shaped steps: a few rows of x against a matrix that
// is read once from memory. Every weight is used by only those few rows, so the product is bound
// by how fast the matrix streams in, and the kernel is built to keep the memory bus busy.

#include "common.cuh"
#include "kvbound/error.h"

namespace kvb::kernels {
namespace {

// Output channels per block: one warp each.
constexpr int kChannelsPerBlock = kBlockWarps;

// Adds the dot product of 8 weights with 8 inputs of each row of x to sums.
template <int kRows>
__device__ inline void accumulate(uint4 load, const uint4* x, std::size_t row_loads, int n_rows,
                                  float* sums) {
  const __half2* w2 = reinterpret_cast<const __half2*>(&load);
  float2 w[4];
  for (int i = 0; i < 4; ++i) w[i] = __half22float2(w2[i]);
  for (int r = 0; r < kRows; ++r) {
    if (r >= n_rows) break;
    const uint4 in = __ldg(x + r * row_loads);
    const __half2* x2 = reinterpret_cast<const __half2*>(&in);
    float sum = sums[r];
    for (int i = 0; i < 4; ++i) {
      const float2 v = __half22float2(x2[i]);
      sum = fmaf(w[i].x, v.x, sum);
      sum = fmaf(w[i].y, v.y, sum);
    }
    sums[r] = sum;
  }
}

// One warp per output channel. Lane l reads 16-byte loads l, l + 32, ... of the channel's row, so
// each trip has the warp read one contiguous 512-byte stretch. Memory parallelism comes from
// resident warps rather than from loads in flight per lane: keeping several loads in flight per
// lane costs registers, hence occupancy, and measured slower on the RTX 4070. Weights are read
// with a streaming hint: each is used once, and keeping them out of L1 leaves it to the rows of
// x, which every warp of the block reads. kRows bounds the rows of x one launch takes, so the
// per-row sums stay in registers. A bias, when there is one, joins each sum as it is written.
template <int kRows, bool kF32Out>
__global__ void linear_f16_kernel(const uint4* weight, const __half* bias, const __half* x, void* y,
                                  int n_rows, int n_in, int n_out) {
  const int lane = threadIdx.x % kWarpSize;
  const std::int64_t channel =
      static_cast<std::int64_t>(blockIdx.x) * kChannelsPerBlock + threadIdx.x / kWarpSize;
  if (channel >= n_out) return;
  const int n_loads = n_in / 8;
  const uint4* row = weight + channel * n_loads;
  const uint4* inputs = reinterpret_cast<const uint4*>(x);
  const std::size_t row_loads = static_cast<std::size_t>(n_loads);

  float sums[kRows];
  for (int r = 0; r < kRows; ++r) sums[r] = 0.0f;
  for (int load = lane; load < n_loads; load += kWarpSize) {
    accumulate<kRows>(__ldcs(row + load), inputs + load, row_loads, n_rows, sums);
  }
  for (int r = 0; r < kRows; ++r) {
    if (r >= n_rows) break;
    float sum = warp_sum(sums[r]);
    if (lane != 0) continue;
    if (bias != nullptr) sum += __half2float(bias[channel]);
    const std::size_t at = static_cast<std::size_t>(r) * n_out + channel;
    if constexpr (kF32Out) {
      static_cast<float*>(y)[at] = sum;
    } else {
      static_cast<__half*>(y)[at] = __float2half(sum);
    }
  }
}

template <bool kF32Out>
void launch(const f16* weight, const f16* bias, const f16* x, void* y, int n_rows, int n_in,
            int n_out, cudaStream_t stream) {
  const unsigned grid = static_cast<unsigned>((n_out + kChannelsPerBlock - 1) / kChannelsPerBlock);
  const uint4* w = reinterpret_cast<const uint4*>(weight);
#define KVB_LAUNCH(rows)           \
  linear_f16_kernel<rows, kF32Out> \
      <<<grid, kBlockThreads, 0, stream>>>(w, as_half(bias), as_half(x), y, n_rows, n_in, n_out)
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

void linear_f16(const f16* weight, const f16* bias, const f16* x, void* y, int n_rows, int n_in,
                int n_out, bool f32_out, cudaStream_t stream) {
  if (n_in % 8 != 0) throw Error("linear_f16 needs rows of a multiple of 8 elements");
  const std::size_t out_bytes = f32_out ? sizeof(float) : sizeof(f16);
  // The matrix is read once for every kF16LinearMaxRows rows.
  for (int first = 0; first < n_rows; first += kF16LinearMaxRows) {
    const int rows = std::min(kF16LinearMaxRows, n_rows - first);
    const f16* xs = x + static_cast<std::size_t>(first) * n_in;
    void* ys = static_cast<std::byte*>(y) + static_cast<std::size_t>(first) * n_out * out_bytes;
    if (f32_out) {
      launch<true>(weight, bias, xs, ys, rows, n_in, n_out, stream);
    } else {
      launch<false>(weight, bias, xs, ys, rows, n_in, n_out, stream);
    }
  }
}

}  // namespace kvb::kernels
