// Kernels that reduce across a row: RMSNorm, LayerNorm and argmax with one block per row, and
// the candidates of beam search over a grid of row chunks.

#include <math_constants.h>

#include <climits>

#include "common.cuh"

namespace kvb::kernels {
namespace {

// Each thread squares a strided share of the row, the block sums the shares with warp shuffles,
// and every thread then scales its share by the same factor. With kAdd, the thread first adds y
// to its share of x and stores the FP16 sum at `sum`, which is x itself, and normalizes the sum;
// it rereads only elements it wrote itself.
template <bool kAdd>
__global__ void rms_norm_kernel(const __half* x, const __half* y, __half* sum, const __half* weight,
                                __half* out, int dim, float eps) {
  __shared__ float scratch[kBlockWarps];
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * dim;
  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    float value = __half2float(x[row + d]);
    if constexpr (kAdd) {
      const __half added = __float2half(value + __half2float(y[row + d]));
      sum[row + d] = added;
      value = __half2float(added);
    }
    sum_sq += value * value;
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    out[row + d] = __float2half(__half2float(x[row + d]) * scale * __half2float(weight[d]));
  }
}

// 8-element chunks of a row one thread of rms_norm_wide_kernel holds.
constexpr int kNormChunks = 4;

// The same norm for rows that split into 16-byte chunks, kNormChunks or fewer per thread: each
// thread loads its chunks once, keeps them in registers for the second pass, and reads and
// writes 8 elements per memory access. A decode step normalizes one row per block, so the
// kernel's time is its chain of dependent memory accesses, and this keeps that chain short.
template <bool kAdd>
__global__ void rms_norm_wide_kernel(const uint4* x, const uint4* y, uint4* sum,
                                     const uint4* weight, uint4* out, int dim, float eps) {
  __shared__ float scratch[kBlockWarps];
  const int n_chunks = dim / 8;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * n_chunks;
  float v[kNormChunks][8];
  float sum_sq = 0.0f;
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 in = x[row + i];
    const __half2* h = reinterpret_cast<const __half2*>(&in);
    uint4 added;
    __half2* a = reinterpret_cast<__half2*>(&added);
    const uint4 extra = kAdd ? y[row + i] : uint4{};
    const __half2* e = reinterpret_cast<const __half2*>(&extra);
    for (int j = 0; j < 4; ++j) {
      float2 f = __half22float2(h[j]);
      if constexpr (kAdd) {
        const float2 g = __half22float2(e[j]);
        a[j] = __floats2half2_rn(f.x + g.x, f.y + g.y);
        f = __half22float2(a[j]);
      }
      v[c][2 * j] = f.x;
      v[c][2 * j + 1] = f.y;
      sum_sq += f.x * f.x + f.y * f.y;
    }
    if constexpr (kAdd) sum[row + i] = added;
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 wl = weight[i];
    const __half2* w = reinterpret_cast<const __half2*>(&wl);
    uint4 result;
    __half2* r = reinterpret_cast<__half2*>(&result);
    for (int j = 0; j < 4; ++j) {
      const float2 wf = __half22float2(w[j]);
      r[j] = __floats2half2_rn(v[c][2 * j] * scale * wf.x, v[c][2 * j + 1] * scale * wf.y);
    }
    out[row + i] = result;
  }
}

bool fits_wide_norm(int dim) { return dim % 8 == 0 && dim / 8 <= kNormChunks * kBlockThreads; }

template <bool kAdd>
void launch_norm(const f16* x, const f16* y, f16* sum, const f16* weight, f16* out, int n_rows,
                 int dim, float eps, cudaStream_t stream) {
  if (fits_wide_norm(dim)) {
    rms_norm_wide_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(x), reinterpret_cast<const uint4*>(y),
        reinterpret_cast<uint4*>(sum), reinterpret_cast<const uint4*>(weight),
        reinterpret_cast<uint4*>(out), dim, eps);
  } else {
    rms_norm_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        as_half(x), as_half(y), as_half(sum), as_half(weight), as_half(out), dim, eps);
  }
  KVB_CUDA(cudaGetLastError());
}

// LayerNorm in the shape of rms_norm_kernel, with one more pass: the block sums the row for its
// mean, then the squared distances from the mean for its variance, and every thread writes its
// share shifted, scaled, and with the bias added. Taking the variance about the mean, not as
// mean(x^2) - mean^2, keeps its digits when the mean is large against the spread.
template <bool kAdd>
__global__ void layer_norm_kernel(const __half* x, const __half* y, __half* sum,
                                  const __half* weight, const __half* bias, __half* out, int dim,
                                  float eps) {
  __shared__ float scratch[kBlockWarps];
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * dim;
  float total = 0.0f;
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    float value = __half2float(x[row + d]);
    if constexpr (kAdd) {
      const __half added = __float2half(value + __half2float(y[row + d]));
      sum[row + d] = added;
      value = __half2float(added);
    }
    total += value;
  }
  const float mean = block_sum(total, scratch) / static_cast<float>(dim);
  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    const float centered = __half2float(x[row + d]) - mean;
    sum_sq += centered * centered;
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    const float centered = __half2float(x[row + d]) - mean;
    out[row + d] = __float2half(centered * scale * __half2float(weight[d]) + __half2float(bias[d]));
  }
}

// LayerNorm in the shape of rms_norm_wide_kernel: a thread's chunks stay in registers through
// all three passes.
template <bool kAdd>
__global__ void layer_norm_wide_kernel(const uint4* x, const uint4* y, uint4* sum,
                                       const uint4* weight, const uint4* bias, uint4* out, int dim,
                                       float eps) {
  __shared__ float scratch[kBlockWarps];
  const int n_chunks = dim / 8;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * n_chunks;
  float v[kNormChunks][8];
  float total = 0.0f;
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 in = x[row + i];
    const __half2* h = reinterpret_cast<const __half2*>(&in);
    uint4 added;
    __half2* a = reinterpret_cast<__half2*>(&added);
    const uint4 extra = kAdd ? y[row + i] : uint4{};
    const __half2* e = reinterpret_cast<const __half2*>(&extra);
    for (int j = 0; j < 4; ++j) {
      float2 f = __half22float2(h[j]);
      if constexpr (kAdd) {
        const float2 g = __half22float2(e[j]);
        a[j] = __floats2half2_rn(f.x + g.x, f.y + g.y);
        f = __half22float2(a[j]);
      }
      v[c][2 * j] = f.x;
      v[c][2 * j + 1] = f.y;
      total += f.x + f.y;
    }
    if constexpr (kAdd) sum[row + i] = added;
  }
  const float mean = block_sum(total, scratch) / static_cast<float>(dim);
  float sum_sq = 0.0f;
  for (int c = 0; c < kNormChunks; ++c) {
    if (threadIdx.x + c * kBlockThreads >= n_chunks) break;
    for (int j = 0; j < 8; ++j) {
      v[c][j] -= mean;
      sum_sq += v[c][j] * v[c][j];
    }
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 wl = weight[i], bl = bias[i];
    const __half2* w = reinterpret_cast<const __half2*>(&wl);
    const __half2* b = reinterpret_cast<const __half2*>(&bl);
    uint4 result;
    __half2* r = reinterpret_cast<__half2*>(&result);
    for (int j = 0; j < 4; ++j) {
      const float2 wf = __half22float2(w[j]), bf = __half22float2(b[j]);
      r[j] = __floats2half2_rn(v[c][2 * j] * scale * wf.x + bf.x,
                               v[c][2 * j + 1] * scale * wf.y + bf.y);
    }
    out[row + i] = result;
  }
}

template <bool kAdd>
void launch_layer_norm(const f16* x, const f16* y, f16* sum, const f16* weight, const f16* bias,
                       f16* out, int n_rows, int dim, float eps, cudaStream_t stream) {
  if (fits_wide_norm(dim)) {
    layer_norm_wide_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(x), reinterpret_cast<const uint4*>(y),
        reinterpret_cast<uint4*>(sum), reinterpret_cast<const uint4*>(weight),
        reinterpret_cast<const uint4*>(bias), reinterpret_cast<uint4*>(out), dim, eps);
  } else {
    layer_norm_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        as_half(x), as_half(y), as_half(sum), as_half(weight), as_half(bias), as_half(out), dim,
        eps);
  }
  KVB_CUDA(cudaGetLastError());
}

struct Best {
  float value;
  int index;
};

__device__ inline Best better(Best a, Best b) {
  return (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
}

__global__ void argmax_kernel(const float* logits, std::int32_t* out, int n_cols) {
  __shared__ Best warp_best[kBlockWarps];
  const float* row = logits + static_cast<std::size_t>(blockIdx.x) * n_cols;
  const int tid = threadIdx.x;
  // Threads past the end of a short row hold a candidate that loses to any real column.
  Best best{-CUDART_INF_F, INT_MAX};
  if (tid < n_cols) best = {row[tid], tid};
  for (int c = tid + kBlockThreads; c < n_cols; c += kBlockThreads) {
    if (row[c] > best.value) best = {row[c], c};
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const Best other{__shfl_xor_sync(0xffffffffu, best.value, offset),
                     __shfl_xor_sync(0xffffffffu, best.index, offset)};
    best = better(best, other);
  }
  if ((tid & (kWarpSize - 1)) == 0) warp_best[tid / kWarpSize] = best;
  __syncthreads();
  if (tid == 0) {
    for (int warp = 1; warp < kBlockWarps; ++warp) best = better(best, warp_best[warp]);
    out[blockIdx.x] = best.index;
  }
}

// --- Beam candidates -------------------------------------------------------------------------
//
// Four passes over grids of (chunk, row): each chunk's maximum; each chunk's sum of
// exp(x - row max) in FP64; each chunk's k best log-probabilities; then each row's k best of its
// chunks' candidates. A pass reads only what earlier launches wrote, so the order of every sum is
// fixed by the launch shape alone.

// Logits of one row one block covers in the first three passes: kBeamPerThread per thread.
constexpr int kBeamPerThread = 8;
constexpr int kBeamChunk = kBeamPerThread * kBlockThreads;

int beam_chunks(int vocab) { return (vocab + kBeamChunk - 1) / kBeamChunk; }

// A candidate ranked as append_beam_candidates() ranks them: log-probability descending, then
// token ascending. {-inf, INT_MAX} stands for no candidate and is behind every finite one.
struct Ranked {
  float log_prob;
  int token;
};

__device__ inline bool ahead(Ranked a, Ranked b) {
  return a.log_prob > b.log_prob || (a.log_prob == b.log_prob && a.token < b.token);
}

// Whether c may be picked after `last`: it is finite and behind `last`, or nothing is picked yet.
__device__ inline bool eligible(Ranked c, Ranked last, bool first) {
  return isfinite(c.log_prob) && (first || ahead(last, c));
}

// The best of every thread's v, returned to every thread. scratch holds one Ranked per warp.
__device__ Ranked block_best(Ranked v, Ranked* scratch) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const Ranked other{__shfl_xor_sync(0xffffffffu, v.log_prob, offset),
                       __shfl_xor_sync(0xffffffffu, v.token, offset)};
    if (ahead(other, v)) v = other;
  }
  if ((threadIdx.x & (kWarpSize - 1)) == 0) scratch[threadIdx.x / kWarpSize] = v;
  __syncthreads();
  Ranked best = scratch[0];
  for (int warp = 1; warp < kBlockWarps; ++warp) {
    if (ahead(scratch[warp], best)) best = scratch[warp];
  }
  __syncthreads();  // every thread has read scratch before the next call writes it
  return best;
}

// Sum over the block in FP64, returned to every thread, in an order fixed by the thread layout.
__device__ double block_sum_f64(double v, double* scratch) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  if ((threadIdx.x & (kWarpSize - 1)) == 0) scratch[threadIdx.x / kWarpSize] = v;
  __syncthreads();
  double total = 0.0;
  for (int warp = 0; warp < kBlockWarps; ++warp) total += scratch[warp];
  __syncthreads();
  return total;
}

// chunk_max[row * n_chunks + chunk] = the largest logit of the chunk.
__global__ void beam_max_kernel(const float* logits, int vocab, float* chunk_max) {
  __shared__ float warp_max[kBlockWarps];
  const int chunk = blockIdx.x, n_chunks = gridDim.x, row = blockIdx.y;
  const float* x = logits + static_cast<std::size_t>(row) * vocab;
  float best = -CUDART_INF_F;
  for (int i = 0; i < kBeamPerThread; ++i) {
    const int token = chunk * kBeamChunk + i * kBlockThreads + threadIdx.x;
    if (token < vocab) best = fmaxf(best, x[token]);
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, offset));
  }
  if ((threadIdx.x & (kWarpSize - 1)) == 0) warp_max[threadIdx.x / kWarpSize] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int warp = 1; warp < kBlockWarps; ++warp) best = fmaxf(best, warp_max[warp]);
    chunk_max[row * n_chunks + chunk] = best;
  }
}

// The row's largest logit, from its chunks' maxima.
__device__ inline float row_max(const float* chunk_max, int row, int n_chunks) {
  float top = -CUDART_INF_F;
  for (int c = 0; c < n_chunks; ++c) top = fmaxf(top, chunk_max[row * n_chunks + c]);
  return top;
}

// chunk_sum[row * n_chunks + chunk] = the chunk's sum of exp(x - row max), in FP64.
__global__ void beam_sum_kernel(const float* logits, int vocab, const float* chunk_max,
                                double* chunk_sum) {
  __shared__ double warp_sum[kBlockWarps];
  __shared__ float shared_top;
  const int chunk = blockIdx.x, n_chunks = gridDim.x, row = blockIdx.y;
  const float* x = logits + static_cast<std::size_t>(row) * vocab;
  if (threadIdx.x == 0) shared_top = row_max(chunk_max, row, n_chunks);
  __syncthreads();
  const double top = shared_top;
  double total = 0.0;
  for (int i = 0; i < kBeamPerThread; ++i) {
    const int token = chunk * kBeamChunk + i * kBlockThreads + threadIdx.x;
    if (token < vocab) total += exp(static_cast<double>(x[token]) - top);
  }
  total = block_sum_f64(total, warp_sum);
  if (threadIdx.x == 0) chunk_sum[row * n_chunks + chunk] = total;
}

// chunk_best[(row * n_chunks + chunk) * k + r] = the chunk's r-th best candidate, or no
// candidate when it has fewer than r + 1.
__global__ void beam_chunk_best_kernel(const float* logits, const StepRow* rows,
                                       const std::int32_t* out_rows, int vocab,
                                       const std::uint8_t* suppress, int begin_pos, int k,
                                       const float* chunk_max, const double* chunk_sum,
                                       Ranked* chunk_best) {
  __shared__ Ranked warp_best[kBlockWarps];
  __shared__ double shared_log_total;
  const int chunk = blockIdx.x, n_chunks = gridDim.x, row = blockIdx.y;
  const float* x = logits + static_cast<std::size_t>(row) * vocab;
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (int c = 0; c < n_chunks; ++c) total += chunk_sum[row * n_chunks + c];
    shared_log_total = static_cast<double>(row_max(chunk_max, row, n_chunks)) + log(total);
  }
  __syncthreads();
  const double log_total = shared_log_total;
  const bool at_begin = rows[out_rows[row]].position == begin_pos;
  const std::uint8_t skip = kSuppressAlways | (at_begin ? kSuppressBegin : 0);

  Ranked mine[kBeamPerThread];
  for (int i = 0; i < kBeamPerThread; ++i) {
    const int token = chunk * kBeamChunk + i * kBlockThreads + threadIdx.x;
    mine[i] = {-CUDART_INF_F, INT_MAX};
    if (token < vocab && (suppress[token] & skip) == 0) {
      mine[i] = {static_cast<float>(static_cast<double>(x[token]) - log_total), token};
    }
  }
  Ranked* out = chunk_best + static_cast<std::size_t>(row * n_chunks + chunk) * k;
  Ranked last{-CUDART_INF_F, INT_MAX};
  for (int r = 0; r < k; ++r) {
    Ranked best{-CUDART_INF_F, INT_MAX};
    for (int i = 0; i < kBeamPerThread; ++i) {
      if (eligible(mine[i], last, r == 0) && ahead(mine[i], best)) best = mine[i];
    }
    best = block_best(best, warp_best);
    if (threadIdx.x == 0) out[r] = best;
    last = best;
  }
}

// out[row * k + r] = the row's r-th best candidate over its chunks', padded with {-1, -inf}.
__global__ void beam_merge_kernel(const Ranked* chunk_best, int n_chunks, int k,
                                  BeamCandidate* out) {
  __shared__ Ranked warp_best[kBlockWarps];
  const int row = blockIdx.x;
  const Ranked* candidates = chunk_best + static_cast<std::size_t>(row) * n_chunks * k;
  const int n = n_chunks * k;
  Ranked last{-CUDART_INF_F, INT_MAX};
  for (int r = 0; r < k; ++r) {
    Ranked best{-CUDART_INF_F, INT_MAX};
    for (int i = threadIdx.x; i < n; i += kBlockThreads) {
      if (eligible(candidates[i], last, r == 0) && ahead(candidates[i], best)) best = candidates[i];
    }
    best = block_best(best, warp_best);
    if (threadIdx.x == 0) {
      out[row * k + r] = isfinite(best.log_prob) ? BeamCandidate{best.token, best.log_prob}
                                                 : BeamCandidate{-1, -CUDART_INF_F};
    }
    last = best;
  }
}

}  // namespace

std::size_t beam_candidates_scratch_bytes(int n_out, int vocab, int k) {
  const std::size_t n = static_cast<std::size_t>(n_out) * beam_chunks(vocab);
  return n * (sizeof(double) + sizeof(float) + static_cast<std::size_t>(k) * sizeof(Ranked));
}

void beam_candidates(const float* logits, const StepRow* rows, const std::int32_t* out_rows,
                     int n_out, int vocab, const std::uint8_t* suppress, int begin_pos, int k,
                     void* scratch, BeamCandidate* out, cudaStream_t stream) {
  if (n_out <= 0 || vocab <= 0 || k <= 0) throw Error("beam candidates need rows, logits and k");
  const int n_chunks = beam_chunks(vocab);
  const std::size_t n = static_cast<std::size_t>(n_out) * n_chunks;
  // The FP64 sums first, so every part stays aligned to its element.
  double* chunk_sum = static_cast<double*>(scratch);
  float* chunk_max = reinterpret_cast<float*>(chunk_sum + n);
  Ranked* chunk_best = reinterpret_cast<Ranked*>(chunk_max + n);
  const dim3 grid(n_chunks, n_out);
  beam_max_kernel<<<grid, kBlockThreads, 0, stream>>>(logits, vocab, chunk_max);
  KVB_CUDA(cudaGetLastError());
  beam_sum_kernel<<<grid, kBlockThreads, 0, stream>>>(logits, vocab, chunk_max, chunk_sum);
  KVB_CUDA(cudaGetLastError());
  beam_chunk_best_kernel<<<grid, kBlockThreads, 0, stream>>>(
      logits, rows, out_rows, vocab, suppress, begin_pos, k, chunk_max, chunk_sum, chunk_best);
  KVB_CUDA(cudaGetLastError());
  beam_merge_kernel<<<n_out, kBlockThreads, 0, stream>>>(chunk_best, n_chunks, k, out);
  KVB_CUDA(cudaGetLastError());
}

void rms_norm(const f16* x, const f16* weight, f16* out, int n_rows, int dim, float eps,
              cudaStream_t stream) {
  launch_norm<false>(x, nullptr, nullptr, weight, out, n_rows, dim, eps, stream);
}

void add_rms_norm(f16* x, const f16* y, const f16* weight, f16* out, int n_rows, int dim, float eps,
                  cudaStream_t stream) {
  launch_norm<true>(x, y, x, weight, out, n_rows, dim, eps, stream);
}

void layer_norm(const f16* x, const f16* weight, const f16* bias, f16* out, int n_rows, int dim,
                float eps, cudaStream_t stream) {
  launch_layer_norm<false>(x, nullptr, nullptr, weight, bias, out, n_rows, dim, eps, stream);
}

void add_layer_norm(f16* x, const f16* y, const f16* weight, const f16* bias, f16* out, int n_rows,
                    int dim, float eps, cudaStream_t stream) {
  launch_layer_norm<true>(x, y, x, weight, bias, out, n_rows, dim, eps, stream);
}

void argmax(const float* logits, std::int32_t* out, int n_rows, int n_cols, cudaStream_t stream) {
  argmax_kernel<<<n_rows, kBlockThreads, 0, stream>>>(logits, out, n_cols);
  KVB_CUDA(cudaGetLastError());
}

}  // namespace kvb::kernels
