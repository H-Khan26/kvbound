// Kernels with one thread per output element: embedding lookup, RoPE, the SwiGLU gate, GELU, the
// residual, position and bias adds, the row gather, the frames of a convolution, logit
// suppression, and the load-time weight conversions.

#include <math_constants.h>

#include "common.cuh"
#include "kvbound/error.h"

namespace kvb::kernels {
namespace {

__device__ inline std::size_t global_index() {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__global__ void embedding_kernel(const __half* table, const StepRow* rows, __half* out, int dim,
                                 std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const std::size_t token = i / dim;
  out[i] = table[static_cast<std::size_t>(rows[token].token_id) * dim + i % dim];
}

// One thread per rotated pair. HF Llama pairs dimension j with j + head_dim/2.
__global__ void rope_kernel(__half* x, const StepRow* rows, const float2* cos_sin, int n_heads,
                            int head_dim, std::size_t n_pairs) {
  const std::size_t i = global_index();
  if (i >= n_pairs) return;
  const int half_dim = head_dim / 2;
  const int j = static_cast<int>(i % half_dim);
  const std::size_t head = i / half_dim;  // token * n_heads + head
  const std::size_t token = head / n_heads;
  const float2 cs = cos_sin[static_cast<std::size_t>(rows[token].position) * half_dim + j];
  __half* vec = x + head * head_dim;
  const float a = __half2float(vec[j]);
  const float b = __half2float(vec[j + half_dim]);
  vec[j] = __float2half(a * cs.x - b * cs.y);
  vec[j + half_dim] = __float2half(b * cs.x + a * cs.y);
}

__global__ void silu_mul_kernel(const __half* gate, const __half* up, __half* out, int dim,
                                int in_stride, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const std::size_t in = i / dim * in_stride + i % dim;
  const float g = __half2float(gate[in]);
  out[i] = __float2half(g / (1.0f + __expf(-g)) * __half2float(up[in]));
}

__global__ void add_kernel(__half* x, const __half* y, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

__global__ void gather_rows_kernel(const __half* x, const std::int32_t* out_rows, __half* out,
                                   int dim, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  out[i] = x[static_cast<std::size_t>(out_rows[i / dim]) * dim + i % dim];
}

__global__ void gelu_kernel(const __half* x, __half* out, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const float v = __half2float(x[i]);
  // 1 + erf(z) is erfc(-z), which keeps its digits where erf(z) nears -1.
  out[i] = __float2half(0.5f * v * erfcf(-v * CUDART_SQRT_HALF_F));
}

__global__ void add_positions_kernel(const __half* x, const __half* table, const StepRow* rows,
                                     __half* out, int dim, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const std::size_t position = static_cast<std::size_t>(rows[i / dim].position);
  out[i] = __float2half(__half2float(x[i]) + __half2float(table[position * dim + i % dim]));
}

// One thread per output element: column 3 c + k of output frame f is channel c of input frame
// f * stride + k - 1.
__global__ void unfold3_kernel(const __half* x, __half* out, int in_frames, int out_frames,
                               int channels, int stride, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const int col = static_cast<int>(i % (3 * channels));
  const std::size_t row = i / (3 * channels);
  const std::size_t seq = row / out_frames;
  const int frame = static_cast<int>(row % out_frames) * stride + col % 3 - 1;
  const bool inside = frame >= 0 && frame < in_frames;
  out[i] = inside ? x[(seq * in_frames + frame) * channels + col / 3] : __float2half(0.0f);
}

template <bool kF32>
__global__ void add_bias_kernel(void* y, const __half* bias, int n_out, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const float b = __half2float(bias[i % n_out]);
  if constexpr (kF32) {
    static_cast<float*>(y)[i] += b;
  } else {
    __half* out = static_cast<__half*>(y);
    out[i] = __float2half(__half2float(out[i]) + b);
  }
}

// One thread per (output row, list entry): entries below n_always come from `always`, the rest
// from `begin` and apply only to a row at begin_pos.
__global__ void suppress_logits_kernel(float* logits, const StepRow* rows,
                                       const std::int32_t* out_rows, int vocab,
                                       const std::int32_t* always, int n_always,
                                       const std::int32_t* begin, int n_begin, int begin_pos,
                                       std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const int entry = static_cast<int>(i % (n_always + n_begin));
  const std::size_t out_row = i / (n_always + n_begin);
  float* row = logits + out_row * vocab;
  if (entry < n_always) {
    row[always[entry]] = -CUDART_INF_F;
  } else if (rows[out_rows[out_row]].position == begin_pos) {
    row[begin[entry - n_always]] = -CUDART_INF_F;
  }
}

// bfloat16 is the top half of a float, so widening is exact; narrowing to FP16 rounds.
__global__ void bf16_to_f16_kernel(std::uint16_t* data, std::size_t n) {
  const std::size_t i = global_index();
  if (i >= n) return;
  const std::uint32_t bits = static_cast<std::uint32_t>(data[i]) << 16;
  float value;
  memcpy(&value, &bits, sizeof(value));
  reinterpret_cast<__half*>(data)[i] = __float2half_rn(value);
}

__global__ void f32_to_f16_kernel(const float* src, __half* dst, std::size_t n) {
  const std::size_t i = global_index();
  if (i < n) dst[i] = __float2half_rn(src[i]);
}

__global__ void count_non_finite_kernel(const std::uint16_t* data, std::size_t n,
                                        unsigned long long* count) {
  const std::size_t i = global_index();
  // An all-ones exponent field marks an infinity or a NaN.
  if (i < n && (data[i] & 0x7C00u) == 0x7C00u) atomicAdd(count, 1ull);
}

}  // namespace

void embedding(const f16* table, const StepRow* rows, f16* out, int n_tokens, int dim,
               cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * dim;
  embedding_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(table), rows, as_half(out),
                                                                dim, n);
  KVB_CUDA(cudaGetLastError());
}

void rope(f16* x, const StepRow* rows, const float* cos_sin, int n_tokens, int n_heads,
          int head_dim, cudaStream_t stream) {
  const std::size_t n_pairs = static_cast<std::size_t>(n_tokens) * n_heads * (head_dim / 2);
  rope_kernel<<<blocks_for(n_pairs), kBlockThreads, 0, stream>>>(
      as_half(x), rows, reinterpret_cast<const float2*>(cos_sin), n_heads, head_dim, n_pairs);
  KVB_CUDA(cudaGetLastError());
}

void silu_mul(const f16* gate, const f16* up, f16* out, int n_rows, int dim, int in_stride,
              cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_rows) * dim;
  silu_mul_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(gate), as_half(up),
                                                               as_half(out), dim, in_stride, n);
  KVB_CUDA(cudaGetLastError());
}

void add(f16* x, const f16* y, std::size_t n, cudaStream_t stream) {
  add_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(x), as_half(y), n);
  KVB_CUDA(cudaGetLastError());
}

void gather_rows(const f16* x, const std::int32_t* out_rows, f16* out, int n_out, int dim,
                 cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_out) * dim;
  gather_rows_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(x), out_rows,
                                                                  as_half(out), dim, n);
  KVB_CUDA(cudaGetLastError());
}

void gelu(const f16* x, f16* out, std::size_t n, cudaStream_t stream) {
  gelu_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(x), as_half(out), n);
  KVB_CUDA(cudaGetLastError());
}

void add_positions(const f16* x, const f16* table, const StepRow* rows, f16* out, int n_rows,
                   int dim, cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_rows) * dim;
  add_positions_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(x), as_half(table),
                                                                    rows, as_half(out), dim, n);
  KVB_CUDA(cudaGetLastError());
}

void unfold3(const f16* x, f16* out, int n_seqs, int in_frames, int channels, int stride,
             cudaStream_t stream) {
  if (in_frames < 1 || channels < 1 || stride < 1) {
    throw Error("unfold3 needs at least one frame, one channel and a stride of 1 or more");
  }
  const int out_frames = (in_frames - 1) / stride + 1;
  const std::size_t n = static_cast<std::size_t>(n_seqs) * out_frames * 3 * channels;
  unfold3_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(x), as_half(out), in_frames,
                                                              out_frames, channels, stride, n);
  KVB_CUDA(cudaGetLastError());
}

void add_bias(void* y, const f16* bias, int n_rows, int n_out, bool f32, cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_rows) * n_out;
  if (f32) {
    add_bias_kernel<true><<<blocks_for(n), kBlockThreads, 0, stream>>>(y, as_half(bias), n_out, n);
  } else {
    add_bias_kernel<false><<<blocks_for(n), kBlockThreads, 0, stream>>>(y, as_half(bias), n_out, n);
  }
  KVB_CUDA(cudaGetLastError());
}

void suppress_logits(float* logits, const StepRow* rows, const std::int32_t* out_rows, int n_out,
                     int vocab, const std::int32_t* always, int n_always, const std::int32_t* begin,
                     int n_begin, int begin_pos, cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_out) * (n_always + n_begin);
  if (n == 0) return;
  suppress_logits_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(
      logits, rows, out_rows, vocab, always, n_always, begin, n_begin, begin_pos, n);
  KVB_CUDA(cudaGetLastError());
}

void bf16_to_f16(f16* data, std::size_t n, cudaStream_t stream) {
  bf16_to_f16_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(data, n);
  KVB_CUDA(cudaGetLastError());
}

void f32_to_f16(const float* src, f16* dst, std::size_t n, cudaStream_t stream) {
  f32_to_f16_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(src, as_half(dst), n);
  KVB_CUDA(cudaGetLastError());
}

std::size_t count_non_finite(const f16* data, std::size_t n, cudaStream_t stream) {
  cuda::DeviceBuffer count(sizeof(unsigned long long));
  KVB_CUDA(cudaMemsetAsync(count.as(), 0, count.bytes(), stream));
  count_non_finite_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(
      data, n, count.as<unsigned long long>());
  KVB_CUDA(cudaGetLastError());
  unsigned long long result = 0;
  KVB_CUDA(cudaMemcpyAsync(&result, count.as(), sizeof(result), cudaMemcpyDeviceToHost, stream));
  KVB_CUDA(cudaStreamSynchronize(stream));
  return static_cast<std::size_t>(result);
}

}  // namespace kvb::kernels
