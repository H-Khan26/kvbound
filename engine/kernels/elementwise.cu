// Kernels with one thread per output element: embedding lookup, RoPE, the SwiGLU gate, the
// residual add, the row gather, and the load-time weight conversions.

#include "common.cuh"

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
