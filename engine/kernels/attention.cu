// KV-cache kernels: storing a step's keys and values, and attention of each query over the
// cached keys of its sequence. Both address the cache only through KvView, so the pool layout
// and the block size are decided by the caller.

#include <math_constants.h>

#include "common.cuh"
#include "kvbound/error.h"

namespace kvb::kernels {
namespace {

// Threads per attention block. A thread waits tens to hundreds of nanoseconds for each load it
// depends on, so with few blocks (decode) the kernel is fastest when each block spreads its keys
// over many threads and every thread's chain of loads is short. With many blocks (prefill) the
// GPU is already full and smaller blocks waste fewer idle threads.
constexpr int kAttnMinThreads = 256;
constexpr int kAttnMaxThreads = 1024;

// Maps positions of one sequence and KV head to pool offsets. The offset is of the vector's
// first element; its dimensions are dim_stride apart.
struct SeqKv {
  const std::int32_t* blocks;  // the sequence's row of the block table
  std::int64_t first_block;    // blocks[0], read once: the only block when blocks_per_seq is 1
  std::int64_t head_offset;
  std::int32_t blocks_per_seq;
  std::uint32_t block_size;
  std::int64_t block_stride;
  std::int64_t pos_stride;

  __device__ SeqKv(const KvView& kv, int seq, int kv_head)
      : blocks(kv.block_table + seq * kv.blocks_per_seq),
        first_block(blocks[0]),
        head_offset(kv_head * kv.head_stride),
        blocks_per_seq(kv.blocks_per_seq),
        block_size(static_cast<std::uint32_t>(kv.block_size)),
        block_stride(kv.block_stride),
        pos_stride(kv.pos_stride) {}

  __device__ std::int64_t offset(int pos) const {
    if (blocks_per_seq == 1) return first_block * block_stride + head_offset + pos * pos_stride;
    const std::uint32_t p = static_cast<std::uint32_t>(pos);
    const std::int64_t block = blocks[p / block_size];
    return block * block_stride + head_offset + (p % block_size) * pos_stride;
  }
};

// Dot product of the query with one cached key. Eight loads are issued before any is used, so
// the thread waits for memory once per eight elements instead of once per element.
__device__ inline float key_dot(const float* query, const __half* key, std::int64_t stride,
                                int head_dim) {
  float dot = 0.0f;
  int d = 0;
  for (; d + 8 <= head_dim; d += 8) {
    const __half k0 = key[(d + 0) * stride], k1 = key[(d + 1) * stride];
    const __half k2 = key[(d + 2) * stride], k3 = key[(d + 3) * stride];
    const __half k4 = key[(d + 4) * stride], k5 = key[(d + 5) * stride];
    const __half k6 = key[(d + 6) * stride], k7 = key[(d + 7) * stride];
    dot += query[d + 0] * __half2float(k0) + query[d + 1] * __half2float(k1) +
           query[d + 2] * __half2float(k2) + query[d + 3] * __half2float(k3) +
           query[d + 4] * __half2float(k4) + query[d + 5] * __half2float(k5) +
           query[d + 6] * __half2float(k6) + query[d + 7] * __half2float(k7);
  }
  for (; d < head_dim; ++d) dot += query[d] * __half2float(key[d * stride]);
  return dot;
}

__global__ void kv_write_kernel(const __half* k, const __half* v, const StepRow* rows, KvView kv,
                                int n_kv_heads, int head_dim, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int dim = static_cast<int>(i % head_dim);
  const std::size_t vec = i / head_dim;  // token * n_kv_heads + head
  const int head = static_cast<int>(vec % n_kv_heads);
  const unsigned slot = static_cast<unsigned>(rows[vec / n_kv_heads].kv_slot);
  const std::int64_t block = slot / static_cast<unsigned>(kv.block_size);
  const std::int64_t in_block = slot % static_cast<unsigned>(kv.block_size);
  const std::int64_t dst = block * kv.block_stride + head * kv.head_stride +
                           in_block * kv.pos_stride + dim * kv.dim_stride;
  static_cast<__half*>(kv.k)[dst] = k[i];
  static_cast<__half*>(kv.v)[dst] = v[i];
}

// One block per (query head, token). Shared memory, in floats:
//   [0, head_dim)        the query vector
//   [.., +threads)       per-thread partial outputs
//   [.., +warps)         per-warp reduction scratch
//   [.., +max_ctx)       one score, then one softmax weight, per cached position
//
// Pass 1, scores: one thread per cached position computes that key's dot product.
// Pass 2, softmax: subtracting the largest score before exp() keeps every exponent <= 0.
// Pass 3, values: threads are split into groups of head_dim. Group g handles positions
// g, g + n_groups, ... and thread d of a group accumulates dimension d, so adjacent threads read
// adjacent elements of one value vector.
__global__ void attention_kernel(const __half* q, __half* out, const StepRow* rows, KvView kv,
                                 int n_heads, int n_rep, int head_dim) {
  extern __shared__ float smem[];
  float* query = smem;
  float* partial = query + head_dim;
  const int n_threads = blockDim.x;
  const int n_warps = n_threads / kWarpSize;
  float* scratch = partial + n_threads;
  float* scores = scratch + n_warps;

  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const StepRow row = rows[blockIdx.y];
  const int ctx = row.position + 1;
  const SeqKv cache(kv, row.seq, head / n_rep);
  const std::size_t vec = (static_cast<std::size_t>(blockIdx.y) * n_heads + head) * head_dim;

  if (tid < head_dim) query[tid] = __half2float(q[vec + tid]);
  __syncthreads();

  const __half* keys = static_cast<const __half*>(kv.k);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float max_score = -CUDART_INF_F;
  for (int pos = tid; pos < ctx; pos += n_threads) {
    const float score = key_dot(query, keys + cache.offset(pos), kv.dim_stride, head_dim) * scale;
    scores[pos] = score;
    max_score = fmaxf(max_score, score);
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffffu, max_score, offset));
  }
  if ((tid & (kWarpSize - 1)) == 0) scratch[tid / kWarpSize] = max_score;
  __syncthreads();
  for (int warp = 0; warp < n_warps; ++warp) max_score = fmaxf(max_score, scratch[warp]);
  __syncthreads();

  float weight_sum = 0.0f;
  for (int pos = tid; pos < ctx; pos += n_threads) {
    const float weight = __expf(scores[pos] - max_score);
    scores[pos] = weight;
    weight_sum += weight;
  }
  weight_sum = block_sum(weight_sum, scratch);

  const __half* values = static_cast<const __half*>(kv.v) + (tid % head_dim) * kv.dim_stride;
  const int n_groups = n_threads / head_dim;
  float acc = 0.0f;
  if (tid / head_dim < n_groups) {
    // Four positions per iteration, loaded before they are used, for the same reason as in
    // key_dot.
    int pos = tid / head_dim;
    for (; pos + 3 * n_groups < ctx; pos += 4 * n_groups) {
      const __half v0 = values[cache.offset(pos)];
      const __half v1 = values[cache.offset(pos + n_groups)];
      const __half v2 = values[cache.offset(pos + 2 * n_groups)];
      const __half v3 = values[cache.offset(pos + 3 * n_groups)];
      acc += scores[pos] * __half2float(v0) + scores[pos + n_groups] * __half2float(v1) +
             scores[pos + 2 * n_groups] * __half2float(v2) +
             scores[pos + 3 * n_groups] * __half2float(v3);
    }
    for (; pos < ctx; pos += n_groups) {
      acc += scores[pos] * __half2float(values[cache.offset(pos)]);
    }
  }
  partial[tid] = acc;
  __syncthreads();
  if (tid < head_dim) {
    float sum = 0.0f;
    for (int g = 0; g < n_groups; ++g) sum += partial[g * head_dim + tid];
    out[vec + tid] = __float2half(sum / weight_sum);
  }
}

}  // namespace

void kv_write(const f16* k, const f16* v, const StepRow* rows, const KvView& kv, int n_tokens,
              int n_kv_heads, int head_dim, cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * n_kv_heads * head_dim;
  kv_write_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(k), as_half(v), rows, kv,
                                                               n_kv_heads, head_dim, n);
  KVB_CUDA(cudaGetLastError());
}

void attention(const f16* q, f16* out, const StepRow* rows, const KvView& kv, int n_tokens,
               int n_heads, int n_kv_heads, int head_dim, int max_ctx, cudaStream_t stream) {
  // Threads the device runs at once: larger blocks only help while they all fit.
  static const std::int64_t device_threads = [] {
    int device = 0;
    cudaDeviceProp prop{};
    KVB_CUDA(cudaGetDevice(&device));
    KVB_CUDA(cudaGetDeviceProperties(&prop, device));
    return static_cast<std::int64_t>(prop.multiProcessorCount) * prop.maxThreadsPerMultiProcessor;
  }();
  const std::int64_t n_blocks = static_cast<std::int64_t>(n_heads) * n_tokens;
  int threads = kAttnMaxThreads;
  while (threads > kAttnMinThreads && n_blocks * threads > device_threads) threads /= 2;

  const std::size_t shared_bytes = sizeof(float) * (static_cast<std::size_t>(head_dim) + threads +
                                                    threads / kWarpSize + max_ctx);
  // 48 KiB is the shared memory a block gets without opting in to more.
  if (head_dim > kAttnMinThreads || shared_bytes > 48 * 1024) {
    throw Error("attention supports head_dim up to 256 and about 11,000 positions");
  }
  const dim3 grid(n_heads, n_tokens);
  attention_kernel<<<grid, threads, shared_bytes, stream>>>(
      as_half(q), as_half(out), rows, kv, n_heads, n_heads / n_kv_heads, head_dim);
  KVB_CUDA(cudaGetLastError());
}

}  // namespace kvb::kernels
