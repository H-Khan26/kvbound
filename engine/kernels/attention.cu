// KV-cache kernels: storing a step's keys and values, and attention of each query over the
// cached keys of its sequence. Both address the cache only through KvView, so the pool layout,
// the block size and the element type are decided by the caller.
//
// An INT8 cache holds each vector as integers times one FP16 scale. Writing picks the scale
// from the vector's largest magnitude; attention multiplies a key's score and a value's softmax
// weight by the scale, so the integers are never expanded in memory.

#include <math_constants.h>

#include <type_traits>

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
// Shared memory a block gets without opting in to more.
constexpr std::size_t kSharedBytes = 48 * 1024;

std::size_t attention_shared_bytes(int head_dim, int threads, int max_ctx) {
  return sizeof(float) *
         (static_cast<std::size_t>(head_dim) + threads + threads / kWarpSize + max_ctx);
}

// Maps positions of one sequence and KV head to pool offsets. The offset is of the vector's
// first element; its dimensions are dim_stride apart.
struct SeqKv {
  const std::int32_t* blocks;  // the sequence's row of the block table
  std::int64_t first_block;    // blocks[0], read once: the only block when blocks_per_seq is 1
  std::int64_t head_offset;
  std::int32_t blocks_per_seq;
  std::int32_t block_shift;
  std::uint32_t block_mask;
  std::int64_t block_stride;
  std::int64_t pos_stride;
  std::int64_t scale_head_offset;
  std::int64_t scale_block_stride;

  __device__ SeqKv(const KvView& kv, int seq, int kv_head)
      : blocks(kv.block_table + seq * kv.blocks_per_seq),
        first_block(blocks[0]),
        head_offset(kv_head * kv.head_stride),
        blocks_per_seq(kv.blocks_per_seq),
        block_shift(kv.block_shift),
        block_mask(static_cast<std::uint32_t>(kv.block_size) - 1),
        block_stride(kv.block_stride),
        pos_stride(kv.pos_stride),
        scale_head_offset(kv_head * kv.scale_head_stride),
        scale_block_stride(kv.scale_block_stride) {}

  // Called once per element read. A paged cache splits the position with a shift and a mask,
  // which is why its block size is a power of two: a division here costs more than the load.
  __device__ std::int64_t offset(int pos) const {
    if (blocks_per_seq == 1) return first_block * block_stride + head_offset + pos * pos_stride;
    const std::uint32_t p = static_cast<std::uint32_t>(pos);
    const std::int64_t block = blocks[p >> block_shift];
    return block * block_stride + head_offset + (p & block_mask) * pos_stride;
  }

  // Index of the position's vector in the scale arrays of a quantized cache.
  __device__ std::int64_t scale_index(int pos) const {
    if (blocks_per_seq == 1) return first_block * scale_block_stride + scale_head_offset + pos;
    const std::uint32_t p = static_cast<std::uint32_t>(pos);
    const std::int64_t block = blocks[p >> block_shift];
    return block * scale_block_stride + scale_head_offset + (p & block_mask);
  }
};

__device__ inline float as_float(__half value) { return __half2float(value); }
__device__ inline float as_float(std::int8_t value) { return static_cast<float>(value); }

// Dot product of the query with one cached key. Eight loads are issued before any is used, so
// the thread waits for memory once per eight elements instead of once per element.
template <typename Elem>
__device__ inline float key_dot(const float* query, const Elem* key, std::int64_t stride,
                                int head_dim) {
  float dot = 0.0f;
  int d = 0;
  for (; d + 8 <= head_dim; d += 8) {
    const Elem k0 = key[(d + 0) * stride], k1 = key[(d + 1) * stride];
    const Elem k2 = key[(d + 2) * stride], k3 = key[(d + 3) * stride];
    const Elem k4 = key[(d + 4) * stride], k5 = key[(d + 5) * stride];
    const Elem k6 = key[(d + 6) * stride], k7 = key[(d + 7) * stride];
    dot += query[d + 0] * as_float(k0) + query[d + 1] * as_float(k1) + query[d + 2] * as_float(k2) +
           query[d + 3] * as_float(k3) + query[d + 4] * as_float(k4) + query[d + 5] * as_float(k5) +
           query[d + 6] * as_float(k6) + query[d + 7] * as_float(k7);
  }
  for (; d < head_dim; ++d) dot += query[d] * as_float(key[d * stride]);
  return dot;
}

// The same dot product for a key whose dimensions are adjacent in memory: each load brings in
// eight of them, so a 64-dimension key takes 8 loads instead of 64.
__device__ inline float key_dot_wide(const float* query, const __half* key, int head_dim) {
  const uint4* words = reinterpret_cast<const uint4*>(key);
  float dot = 0.0f;
  for (int d = 0; d < head_dim; d += 8) {
    const uint4 w = words[d / 8];
    dot += query[d + 0] * f16_bits_to_float(w.x) + query[d + 1] * f16_bits_to_float(w.x >> 16) +
           query[d + 2] * f16_bits_to_float(w.y) + query[d + 3] * f16_bits_to_float(w.y >> 16) +
           query[d + 4] * f16_bits_to_float(w.z) + query[d + 5] * f16_bits_to_float(w.z >> 16) +
           query[d + 6] * f16_bits_to_float(w.w) + query[d + 7] * f16_bits_to_float(w.w >> 16);
  }
  return dot;
}

// An INT8 key holds sixteen dimensions per load, so a 64-dimension key takes 4 loads.
__device__ inline float key_dot_wide(const float* query, const std::int8_t* key, int head_dim) {
  const uint4* words = reinterpret_cast<const uint4*>(key);
  float dot = 0.0f;
  for (int d = 0; d < head_dim; d += 16) {
    float k[16];
    unpack_int8(words[d / 16], k);
    for (int i = 0; i < 16; ++i) dot += query[d + i] * k[i];
  }
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

// One warp per stored vector, a token's key or value for one KV head. The warp finds the
// vector's largest magnitude, stores that over 127 as the FP16 scale, and stores every element
// divided by the scale as it reads back, so a rounded scale costs no extra error.
__global__ void kv_write_int8_kernel(const __half* k, const __half* v, const StepRow* rows,
                                     KvView kv, int n_kv_heads, int head_dim,
                                     std::size_t n_vectors) {
  const std::size_t vec =
      static_cast<std::size_t>(blockIdx.x) * kBlockWarps + threadIdx.x / kWarpSize;
  if (vec >= n_vectors) return;
  const int lane = threadIdx.x % kWarpSize;
  const int head = static_cast<int>(vec % n_kv_heads);
  const unsigned slot = static_cast<unsigned>(rows[vec / n_kv_heads].kv_slot);
  const std::int64_t block = slot / static_cast<unsigned>(kv.block_size);
  const std::int64_t in_block = slot % static_cast<unsigned>(kv.block_size);
  const std::int64_t dst =
      block * kv.block_stride + head * kv.head_stride + in_block * kv.pos_stride;
  const std::int64_t scale_at =
      block * kv.scale_block_stride + head * kv.scale_head_stride + in_block;

  for (int which = 0; which < 2; ++which) {
    const __half* src = (which == 0 ? k : v) + vec * head_dim;
    float largest = 0.0f;
    for (int d = lane; d < head_dim; d += kWarpSize) {
      largest = fmaxf(largest, fabsf(__half2float(src[d])));
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
    }
    const __half stored = __float2half(largest / 127.0f);
    const float scale = __half2float(stored);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    std::int8_t* out = static_cast<std::int8_t*>(which == 0 ? kv.k : kv.v) + dst;
    for (int d = lane; d < head_dim; d += kWarpSize) {
      const float q = rintf(__half2float(src[d]) * inverse);
      out[d * kv.dim_stride] = static_cast<std::int8_t>(fminf(fmaxf(q, -127.0f), 127.0f));
    }
    if (lane == 0) static_cast<__half*>(which == 0 ? kv.k_scale : kv.v_scale)[scale_at] = stored;
  }
}

// One block per (query head, token). Shared memory, in floats:
//   [0, head_dim)        the query vector
//   [.., +threads)       per-thread partial outputs
//   [.., +warps)         per-warp reduction scratch
//   [.., +max_ctx)       one score, then one softmax weight, per cached position
//
// Pass 1, scores: one thread per cached position computes that key's dot product, reading the
// key element by element or, with kWideKeys, 16 bytes at a time.
// Pass 2, softmax: subtracting the largest score before exp() keeps every exponent <= 0.
// Pass 3, values: threads are split into groups of head_dim. Group g handles positions
// g, g + n_groups, ... and thread d of a group accumulates dimension d, so adjacent threads read
// adjacent elements of one value vector.
//
// With kInt8 the cache holds integers: a score is multiplied by its key's scale, and each
// softmax weight by its value's scale before pass 3 uses it.
template <bool kWideKeys, bool kInt8>
__global__ void attention_kernel(const __half* q, __half* out, const StepRow* rows, KvView kv,
                                 int n_heads, int n_rep, int head_dim) {
  using Elem = std::conditional_t<kInt8, std::int8_t, __half>;
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

  const Elem* keys = static_cast<const Elem*>(kv.k);
  const __half* key_scales = static_cast<const __half*>(kv.k_scale);
  const __half* value_scales = static_cast<const __half*>(kv.v_scale);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float max_score = -CUDART_INF_F;
  for (int pos = tid; pos < ctx; pos += n_threads) {
    const Elem* key = keys + cache.offset(pos);
    float score = (kWideKeys ? key_dot_wide(query, key, head_dim)
                             : key_dot(query, key, kv.dim_stride, head_dim)) *
                  scale;
    if constexpr (kInt8) score *= __half2float(key_scales[cache.scale_index(pos)]);
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
    weight_sum += weight;
    if constexpr (kInt8) {
      scores[pos] = weight * __half2float(value_scales[cache.scale_index(pos)]);
    } else {
      scores[pos] = weight;
    }
  }
  weight_sum = block_sum(weight_sum, scratch);

  const Elem* values = static_cast<const Elem*>(kv.v) + (tid % head_dim) * kv.dim_stride;
  const int n_groups = n_threads / head_dim;
  float acc = 0.0f;
  if (tid / head_dim < n_groups) {
    // Four positions per iteration, loaded before they are used, for the same reason as in
    // key_dot.
    int pos = tid / head_dim;
    for (; pos + 3 * n_groups < ctx; pos += 4 * n_groups) {
      const Elem v0 = values[cache.offset(pos)];
      const Elem v1 = values[cache.offset(pos + n_groups)];
      const Elem v2 = values[cache.offset(pos + 2 * n_groups)];
      const Elem v3 = values[cache.offset(pos + 3 * n_groups)];
      acc += scores[pos] * as_float(v0) + scores[pos + n_groups] * as_float(v1) +
             scores[pos + 2 * n_groups] * as_float(v2) + scores[pos + 3 * n_groups] * as_float(v3);
    }
    for (; pos < ctx; pos += n_groups) {
      acc += scores[pos] * as_float(values[cache.offset(pos)]);
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
  if (kv.dtype == DType::I8) {
    const std::size_t n_vectors = static_cast<std::size_t>(n_tokens) * n_kv_heads;
    const unsigned grid = static_cast<unsigned>((n_vectors + kBlockWarps - 1) / kBlockWarps);
    kv_write_int8_kernel<<<grid, kBlockThreads, 0, stream>>>(as_half(k), as_half(v), rows, kv,
                                                             n_kv_heads, head_dim, n_vectors);
  } else {
    const std::size_t n = static_cast<std::size_t>(n_tokens) * n_kv_heads * head_dim;
    kv_write_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(as_half(k), as_half(v), rows, kv,
                                                                 n_kv_heads, head_dim, n);
  }
  KVB_CUDA(cudaGetLastError());
}

std::string_view key_loads_name(KeyLoads loads) noexcept {
  return loads == KeyLoads::Wide ? "wide" : "scalar";
}

KeyLoads widest_key_loads(const KvView& kv, int head_dim) noexcept {
  // Elements one 16-byte load holds.
  const int per_load = 16 / static_cast<int>(size_bytes(kv.dtype));
  return kv.dim_stride == 1 && head_dim % per_load == 0 ? KeyLoads::Wide : KeyLoads::Scalar;
}

int attention_max_positions(int head_dim) {
  if (head_dim > kAttnMinThreads) return 0;
  // The largest block size leaves the least room for scores.
  return static_cast<int>((kSharedBytes - attention_shared_bytes(head_dim, kAttnMaxThreads, 0)) /
                          sizeof(float));
}

void attention(const f16* q, f16* out, const StepRow* rows, const KvView& kv, int n_tokens,
               int n_heads, int n_kv_heads, int head_dim, int max_ctx, KeyLoads loads,
               cudaStream_t stream) {
  if (loads == KeyLoads::Wide && widest_key_loads(kv, head_dim) != KeyLoads::Wide) {
    throw Error("wide key loads need adjacent dimensions and a head_dim that fills 16-byte loads");
  }
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

  if (max_ctx > attention_max_positions(head_dim)) {
    throw Error("attention supports head_dim up to 256 and about 11,000 positions");
  }
  const std::size_t shared_bytes = attention_shared_bytes(head_dim, threads, max_ctx);
  const dim3 grid(n_heads, n_tokens);
  const int n_rep = n_heads / n_kv_heads;
  const bool wide = loads == KeyLoads::Wide;
#define KVB_LAUNCH(wide_keys, int8)                                           \
  attention_kernel<wide_keys, int8><<<grid, threads, shared_bytes, stream>>>( \
      as_half(q), as_half(out), rows, kv, n_heads, n_rep, head_dim)
  if (kv.dtype == DType::I8) {
    wide ? KVB_LAUNCH(true, true) : KVB_LAUNCH(false, true);
  } else {
    wide ? KVB_LAUNCH(true, false) : KVB_LAUNCH(false, false);
  }
#undef KVB_LAUNCH
  KVB_CUDA(cudaGetLastError());
}

}  // namespace kvb::kernels
