// KV-cache kernels: storing a step's keys and values, and attention of each query over the
// cached keys of its sequence. Both address the cache only through KvView, so the pool layout,
// the block size and the element type are decided by the caller.
//
// An INT8 cache holds each vector as integers times one FP16 scale. Writing picks the scale
// from the vector's largest magnitude; attention multiplies a key's score and a value's softmax
// weight by the scale, so the integers are never expanded in memory.

#include <math_constants.h>

#include <cmath>
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

// One thread per element of a token's keys; it stores the matching value element too. Token t's
// keys and values start in_stride elements after those of token t - 1.
__global__ void kv_write_kernel(const __half* k, const __half* v, const StepRow* rows, KvView kv,
                                int n_kv_heads, int head_dim, int in_stride, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int dim = static_cast<int>(i % head_dim);
  const std::size_t vec = i / head_dim;  // token * n_kv_heads + head
  const int head = static_cast<int>(vec % n_kv_heads);
  const std::size_t token = vec / n_kv_heads;
  const std::size_t src = token * in_stride + head * head_dim + dim;
  const unsigned slot = static_cast<unsigned>(rows[token].kv_slot);
  const std::int64_t block = slot / static_cast<unsigned>(kv.block_size);
  const std::int64_t in_block = slot % static_cast<unsigned>(kv.block_size);
  const std::int64_t dst = block * kv.block_stride + head * kv.head_stride +
                           in_block * kv.pos_stride + dim * kv.dim_stride;
  static_cast<__half*>(kv.k)[dst] = k[src];
  static_cast<__half*>(kv.v)[dst] = v[src];
}

// One warp per stored vector, a token's key or value for one KV head. The warp finds the
// vector's largest magnitude, stores that over 127 as the FP16 scale, and stores every element
// divided by the scale as it reads back, so a rounded scale costs no extra error.
__global__ void kv_write_int8_kernel(const __half* k, const __half* v, const StepRow* rows,
                                     KvView kv, int n_kv_heads, int head_dim, int in_stride,
                                     std::size_t n_vectors) {
  const std::size_t vec =
      static_cast<std::size_t>(blockIdx.x) * kBlockWarps + threadIdx.x / kWarpSize;
  if (vec >= n_vectors) return;
  const int lane = threadIdx.x % kWarpSize;
  const int head = static_cast<int>(vec % n_kv_heads);
  const std::size_t token = vec / n_kv_heads;
  const unsigned slot = static_cast<unsigned>(rows[token].kv_slot);
  const std::int64_t block = slot / static_cast<unsigned>(kv.block_size);
  const std::int64_t in_block = slot % static_cast<unsigned>(kv.block_size);
  const std::int64_t dst =
      block * kv.block_stride + head * kv.head_stride + in_block * kv.pos_stride;
  const std::int64_t scale_at =
      block * kv.scale_block_stride + head * kv.scale_head_stride + in_block;

  for (int which = 0; which < 2; ++which) {
    const __half* src = (which == 0 ? k : v) + token * in_stride + head * head_dim;
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
//
// A row attends the positions up to its own, or with fixed_ctx > 0 the first fixed_ctx.
template <bool kWideKeys, bool kInt8>
__global__ void attention_kernel(const __half* q, __half* out, const StepRow* rows, KvView kv,
                                 int n_heads, int n_rep, int head_dim, int fixed_ctx) {
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
  const int ctx = fixed_ctx > 0 ? fixed_ctx : row.position + 1;
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

// --- Prefill attention on tensor cores -------------------------------------------------------
//
// Flash attention: a block takes one tile of up to kAttnTileRows consecutive query rows of one
// sequence and walks that sequence's cache in tiles of kPrefillKeys positions, keeping a running
// maximum and sum per query row so no row of scores is ever stored. Products are FP16
// mma.sync.m16n8k16 with FP32 accumulators; each warp owns 16 query rows of one head.

// Cached positions per shared-memory tile.
constexpr int kPrefillKeys = 64;
// Query rows per warp: the M of the MMA.
constexpr int kWarpRows = 16;
// Warps per block, together covering the tile's rows for one query head.
constexpr int kTileWarps = kAttnTileRows / kWarpRows;
constexpr int kPrefillThreads = kTileWarps * kWarpSize;
// Blocks per SM the register budget is fitted to. At head_dim 64 a block's 32 KB of shared
// memory allows three, and without the bound the INT8 kernel takes enough registers to fit only
// two. At head_dim 128 shared memory (64 KB) allows one, and a tighter bound would spill.
constexpr int prefill_blocks_per_sm(int head_dim) { return head_dim <= 64 ? 3 : 1; }

static_assert(kAttnTileRows == kPrefillKeys, "a query tile and a key tile share one layout");

// Halves of one staged tile (keys, values or one head's queries).
__host__ __device__ constexpr int prefill_tile_halves(int head_dim) {
  return kPrefillKeys * head_dim;
}

// Offset in halves of (row, col) in a staged tile, for col a multiple of 8. ldmatrix reads one
// 16-byte chunk from each of eight consecutive rows; with rows of 128 or 256 bytes those chunks
// would all sit in the same four banks. XOR-ing the chunk index with row % 8 moves each row's
// chunk to a different bank group, at no cost in shared memory (unlike padding the rows).
template <int kHeadDim>
__device__ inline int swizzled(int row, int col) {
  return row * kHeadDim + (((col >> 3) ^ (row & 7)) << 3);
}

// Shared memory: two stages of {keys, values}, double-buffered so one tile loads while the
// previous is multiplied. The queries are staged once, before the loop, in the second stage's
// keys, and live in registers afterwards.
constexpr std::size_t prefill_shared_bytes(int head_dim) {
  return sizeof(__half) * 4 * static_cast<std::size_t>(prefill_tile_halves(head_dim));
}

__device__ inline unsigned shared_address(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// Four 8x8 FP16 matrices from shared memory into MMA fragments; lanes 8i..8i+7 give the row
// addresses of matrix i. The .trans form hands out the transposed matrices.
__device__ inline void ldmatrix_x4(unsigned (&r)[4], const __half* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(shared_address(p)));
}
__device__ inline void ldmatrix_x4_trans(unsigned (&r)[4], const __half* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(shared_address(p)));
}

// c[16x8] += a[16x16] * b[16x8]. Lane l holds rows l/4 and l/4 + 8 of c, columns 2(l%4) and
// 2(l%4) + 1 of each.
__device__ inline void mma_16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2^x in one MUFU instruction, flushing denormal results to zero; exp2f() adds instructions
// around it to get denormals right, which a softmax weight below 2^-126 does not need.
// 2^-inf is 0.
__device__ inline float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// Two floats as an FP16 pair, the first in the low half.
__device__ inline unsigned pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&h);
}

// Copies cached positions start..start + kPrefillKeys - 1 of an FP16 cache into a stage with
// cp.async; positions from n_keys on are zero-filled.
template <int kHeadDim, int kThreads>
__device__ inline void stage_kv_f16(__half* k_dst, __half* v_dst, const KvView& kv,
                                    const SeqKv& cache, int start, int n_keys) {
  constexpr int kChunks = kHeadDim / 8;  // 16-byte loads per vector
  const __half* keys = static_cast<const __half*>(kv.k);
  const __half* values = static_cast<const __half*>(kv.v);
  for (int c = threadIdx.x; c < kPrefillKeys * kChunks; c += kThreads) {
    const int r = c / kChunks, col = (c % kChunks) * 8;
    const bool valid = start + r < n_keys;
    // A zero-filled copy reads nothing, but its address still points at position 0, which every
    // sequence has written.
    const std::int64_t at = cache.offset(valid ? start + r : 0) + col;
    cp_async16(k_dst + swizzled<kHeadDim>(r, col), keys + at, valid);
    cp_async16(v_dst + swizzled<kHeadDim>(r, col), values + at, valid);
  }
}

// One tile of an INT8 cache on its way to shared memory: load() issues the global loads,
// store() expands each element to FP16 as integer times its vector's scale. In between, the
// loads complete while the block multiplies the previous tile.
template <int kHeadDim, int kThreads>
struct Int8Staging {
  static constexpr int kChunks = kHeadDim / 16;  // 16-byte loads per vector
  static constexpr int kTotal = kPrefillKeys * kChunks;
  static constexpr int kPerThread = (kTotal + kThreads - 1) / kThreads;
  uint4 k[kPerThread], v[kPerThread];
  float k_scale[kPerThread], v_scale[kPerThread];

  __device__ void load(const KvView& kv, const SeqKv& cache, int start, int n_keys) {
    const std::int8_t* keys = static_cast<const std::int8_t*>(kv.k);
    const std::int8_t* values = static_cast<const std::int8_t*>(kv.v);
    const __half* key_scales = static_cast<const __half*>(kv.k_scale);
    const __half* value_scales = static_cast<const __half*>(kv.v_scale);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int c = threadIdx.x + i * kThreads;
      const int pos = start + c / kChunks;
      if (c < kTotal && pos < n_keys) {
        const std::int64_t at = cache.offset(pos) + (c % kChunks) * 16;
        const std::int64_t scale_at = cache.scale_index(pos);
        k[i] = *reinterpret_cast<const uint4*>(keys + at);
        v[i] = *reinterpret_cast<const uint4*>(values + at);
        k_scale[i] = __half2float(key_scales[scale_at]);
        v_scale[i] = __half2float(value_scales[scale_at]);
      } else {
        k[i] = v[i] = make_uint4(0, 0, 0, 0);
        k_scale[i] = v_scale[i] = 0.0f;
      }
    }
  }

  // Writes the 16 elements of one load as two chunks, at columns col and col + 8 of row r.
  __device__ static void expand(uint4 load, float scale, __half* dst, int r, int col) {
    float x[16];
    unpack_int8(load, x);
    uint4 lo, hi;
    lo.x = pack_half2(x[0] * scale, x[1] * scale);
    lo.y = pack_half2(x[2] * scale, x[3] * scale);
    lo.z = pack_half2(x[4] * scale, x[5] * scale);
    lo.w = pack_half2(x[6] * scale, x[7] * scale);
    hi.x = pack_half2(x[8] * scale, x[9] * scale);
    hi.y = pack_half2(x[10] * scale, x[11] * scale);
    hi.z = pack_half2(x[12] * scale, x[13] * scale);
    hi.w = pack_half2(x[14] * scale, x[15] * scale);
    *reinterpret_cast<uint4*>(dst + swizzled<kHeadDim>(r, col)) = lo;
    *reinterpret_cast<uint4*>(dst + swizzled<kHeadDim>(r, col + 8)) = hi;
  }

  __device__ void store(__half* k_dst, __half* v_dst) const {
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int c = threadIdx.x + i * kThreads;
      if (c >= kTotal) break;
      const int r = c / kChunks, col = (c % kChunks) * 16;
      expand(k[i], k_scale[i], k_dst, r, col);
      expand(v[i], v_scale[i], v_dst, r, col);
    }
  }
};

// One block per (tile, query head); grid.x walks the tiles from last to first, since in a long
// prompt later tiles see more keys and should start first. Blocks of the query heads that share
// a KV head read the same keys and values, which stay in L2. (Letting one block serve several
// such heads from one staged tile measured slower: its larger register file leaves fewer warps
// per SM, and this kernel is bound by the tensor pipe and latency, not by cache reads.)
//
// Per warp and key tile: S = Q K^T (16 x 64, FP32) from fragments; the scores are scaled into
// the log2 domain, masked causally, and folded into the running max m and sum l of each row;
// the accumulated output is rescaled by exp2(m_old - m_new), and P (rounded to FP16, which is
// exactly the A-fragment layout of the next MMA) times V is added to it. Rows of a tile are
// consecutive positions, so row i of the tile sits at position first_pos + i.
//
// Without kCausal there is no mask: every row attends positions 0..dense_keys - 1, whatever its
// own. Head h of query row r starts at q[r * q_stride + h * kHeadDim].
template <int kHeadDim, bool kInt8, bool kCausal>
__global__ void __launch_bounds__(kPrefillThreads, prefill_blocks_per_sm(kHeadDim))
    attention_prefill_kernel(const __half* q, __half* out, const StepRow* rows,
                             const AttnTile* tiles, int n_tiles, KvView kv, int n_heads, int n_rep,
                             float scale_log2, int q_stride, int dense_keys) {
  constexpr int kThreads = kPrefillThreads;
  constexpr int kTileHalves = prefill_tile_halves(kHeadDim);
  constexpr int kDimTiles = kHeadDim / 8;  // n8 output tiles per row
  constexpr int kKeyTiles = kPrefillKeys / 8;
  extern __shared__ __align__(16) unsigned char prefill_smem[];
  __half* smem = reinterpret_cast<__half*>(prefill_smem);
  __half* q_smem = smem + 2 * kTileHalves;

  const AttnTile tile = tiles[n_tiles - 1 - blockIdx.x];
  const int head = blockIdx.y;
  const int kv_head = head / n_rep;
  const StepRow first = rows[tile.first_row];
  const int first_pos = first.position;
  const int n_keys = kCausal ? first_pos + tile.n_rows : dense_keys;
  const int n_key_tiles = (n_keys + kPrefillKeys - 1) / kPrefillKeys;
  const SeqKv cache(kv, first.seq, kv_head);
  const std::size_t token_stride = static_cast<std::size_t>(n_heads) * kHeadDim;

  // The tile's queries; rows past the tile are zero.
  constexpr int kQueryChunks = kHeadDim / 8;
  for (int c = threadIdx.x; c < kAttnTileRows * kQueryChunks; c += kThreads) {
    const int r = c / kQueryChunks, col = (c % kQueryChunks) * 8;
    const bool valid = r < tile.n_rows;
    const __half* src = q +
                        (tile.first_row + (valid ? r : 0)) * static_cast<std::size_t>(q_stride) +
                        head * kHeadDim + col;
    cp_async16(q_smem + swizzled<kHeadDim>(r, col), src, valid);
  }
  [[maybe_unused]] Int8Staging<kHeadDim, kThreads> staging;
  if constexpr (kInt8) {
    staging.load(kv, cache, 0, n_keys);
    staging.store(smem, smem + kTileHalves);
  } else {
    stage_kv_f16<kHeadDim, kThreads>(smem, smem + kTileHalves, kv, cache, 0, n_keys);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const int warp = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;
  const int warp_row = warp * kWarpRows;  // the warp's first row in the tile
  unsigned qf[kHeadDim / 16][4];
  {
    const int r = warp_row + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = (lane >> 4) * 8;
#pragma unroll
    for (int kk = 0; kk < kHeadDim / 16; ++kk) {
      ldmatrix_x4(qf[kk], q_smem + swizzled<kHeadDim>(r, kk * 16 + col));
    }
  }
  __syncthreads();  // the queries' shared memory belongs to the second stage from here on

  // This thread's rows: g and g + 8 of the warp's 16, at positions row_pos and row_pos + 8.
  const int g = lane >> 2, t = lane & 3;
  const int row_pos = first_pos + warp_row + g;
  const bool warp_active = warp_row < tile.n_rows;
  // The last position any row of the warp sees; later key tiles are skipped.
  const int warp_last =
      kCausal ? first_pos + min(warp_row + kWarpRows, tile.n_rows) - 1 : n_keys - 1;
  float o[kDimTiles][4] = {};
  float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float row_sum[2] = {0.0f, 0.0f};

  for (int kt = 0; kt < n_key_tiles; ++kt) {
    if (kt > 0) {
      cp_async_wait<0>();
      __syncthreads();
    }
    const int next = kt + 1;
    __half* next_k = smem + (next & 1) * 2 * kTileHalves;
    if (next < n_key_tiles) {
      if constexpr (kInt8) {
        staging.load(kv, cache, next * kPrefillKeys, n_keys);
      } else {
        stage_kv_f16<kHeadDim, kThreads>(next_k, next_k + kTileHalves, kv, cache,
                                         next * kPrefillKeys, n_keys);
        cp_async_commit();
      }
    }

    const int key0 = kt * kPrefillKeys;
    if (warp_active && key0 <= warp_last) {
      const __half* ks = smem + (kt & 1) * 2 * kTileHalves;
      const __half* vs = ks + kTileHalves;

      float s[kKeyTiles][4] = {};
      {
        const int key = (lane & 7) + (lane >> 4) * 8, col = ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; ++kk) {
#pragma unroll
          for (int nn = 0; nn < kPrefillKeys / 16; ++nn) {
            unsigned b[4];
            ldmatrix_x4(b, ks + swizzled<kHeadDim>(nn * 16 + key, kk * 16 + col));
            mma_16816(s[2 * nn], qf[kk], b[0], b[1]);
            mma_16816(s[2 * nn + 1], qf[kk], b[2], b[3]);
          }
        }
      }

      // Only a tile that reaches past the warp's first row has keys to mask. With no causal
      // mask, only the last tile has: its zero-filled positions from n_keys on.
      const bool masked =
          kCausal ? key0 + kPrefillKeys - 1 > first_pos + warp_row : key0 + kPrefillKeys > n_keys;
      float tile_max[2] = {row_max[0], row_max[1]};
#pragma unroll
      for (int j = 0; j < kKeyTiles; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int key = key0 + j * 8 + 2 * t + (e & 1);
          if (masked && (kCausal ? key > row_pos + (e >> 1) * 8 : key >= n_keys)) {
            s[j][e] = -CUDART_INF_F;
          }
          tile_max[e >> 1] = fmaxf(tile_max[e >> 1], s[j][e]);
        }
      }
      // Maxima stay in unscaled score units, so each weight is one FMA and one exp2. Every row
      // sees position 0 in the first tile, so the maximum is finite from then on and the exp2
      // of -inf (masked keys, and the first rescale) is 0.
      float alpha[2], shift[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        tile_max[i] = fmaxf(tile_max[i], __shfl_xor_sync(0xffffffffu, tile_max[i], 1));
        tile_max[i] = fmaxf(tile_max[i], __shfl_xor_sync(0xffffffffu, tile_max[i], 2));
        alpha[i] = fast_exp2((row_max[i] - tile_max[i]) * scale_log2);
        row_max[i] = tile_max[i];
        row_sum[i] *= alpha[i];
        shift[i] = -tile_max[i] * scale_log2;
      }
#pragma unroll
      for (int j = 0; j < kKeyTiles; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          s[j][e] = fast_exp2(fmaf(s[j][e], scale_log2, shift[e >> 1]));
          row_sum[e >> 1] += s[j][e];
        }
      }
#pragma unroll
      for (int d = 0; d < kDimTiles; ++d) {
        o[d][0] *= alpha[0];
        o[d][1] *= alpha[0];
        o[d][2] *= alpha[1];
        o[d][3] *= alpha[1];
      }

      {
        const int key = (lane & 7) + ((lane >> 3) & 1) * 8, col = (lane >> 4) * 8;
#pragma unroll
        for (int kc = 0; kc < kPrefillKeys / 16; ++kc) {
          const unsigned p[4] = {
              pack_half2(s[2 * kc][0], s[2 * kc][1]),
              pack_half2(s[2 * kc][2], s[2 * kc][3]),
              pack_half2(s[2 * kc + 1][0], s[2 * kc + 1][1]),
              pack_half2(s[2 * kc + 1][2], s[2 * kc + 1][3]),
          };
#pragma unroll
          for (int dd = 0; dd < kHeadDim / 16; ++dd) {
            unsigned b[4];
            ldmatrix_x4_trans(b, vs + swizzled<kHeadDim>(kc * 16 + key, dd * 16 + col));
            mma_16816(o[2 * dd], p, b[0], b[1]);
            mma_16816(o[2 * dd + 1], p, b[2], b[3]);
          }
        }
      }
    }

    if constexpr (kInt8) {
      if (next < n_key_tiles) staging.store(next_k, next_k + kTileHalves);
    }
  }

  // Each of a row's four lanes summed a quarter of its keys.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    row_sum[i] += __shfl_xor_sync(0xffffffffu, row_sum[i], 1);
    row_sum[i] += __shfl_xor_sync(0xffffffffu, row_sum[i], 2);
  }
  if (!warp_active) return;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = warp_row + g + 8 * i;
    if (r >= tile.n_rows) continue;
    const float inverse = 1.0f / row_sum[i];
    __half* dst = out + (tile.first_row + r) * token_stride + head * kHeadDim + 2 * t;
#pragma unroll
    for (int d = 0; d < kDimTiles; ++d) {
      *reinterpret_cast<unsigned*>(dst + d * 8) =
          pack_half2(o[d][2 * i] * inverse, o[d][2 * i + 1] * inverse);
    }
  }
}

template <int kHeadDim, bool kInt8, bool kCausal>
void launch_attention_prefill(const __half* q, __half* out, const StepRow* rows,
                              const AttnTile* tiles, int n_tiles, const KvView& kv, int n_heads,
                              int n_kv_heads, int q_stride, int dense_keys, cudaStream_t stream) {
  constexpr std::size_t shared_bytes = prefill_shared_bytes(kHeadDim);
  auto* kernel = attention_prefill_kernel<kHeadDim, kInt8, kCausal>;
  if constexpr (shared_bytes > kSharedBytes) {
    // Blocks get more than 48 KB of shared memory (64 KB at head_dim 128) only when the kernel
    // opts in.
    static const bool opted_in = [&] {
      KVB_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_bytes)));
      return true;
    }();
    (void)opted_in;
  }
  // Scores go into exp2(), so the 1/sqrt(head_dim) scale also carries log2(e).
  const float scale_log2 = 1.4426950408889634f / std::sqrt(static_cast<float>(kHeadDim));
  const dim3 grid(n_tiles, n_heads);
  kernel<<<grid, kPrefillThreads, shared_bytes, stream>>>(q, out, rows, tiles, n_tiles, kv, n_heads,
                                                          n_heads / n_kv_heads, scale_log2,
                                                          q_stride, dense_keys);
}

// Launches the kernel for this head_dim and cache element type: causal, or with kCausal false
// over dense_keys positions.
template <bool kCausal>
void dispatch_attention_prefill(const f16* q, f16* out, const StepRow* rows, const AttnTile* tiles,
                                int n_tiles, const KvView& kv, int n_heads, int n_kv_heads,
                                int head_dim, int q_stride, int dense_keys, cudaStream_t stream) {
  const bool int8 = kv.dtype == DType::I8;
#define KVB_LAUNCH(dim, quantized)                                                              \
  launch_attention_prefill<dim, quantized, kCausal>(as_half(q), as_half(out), rows, tiles,      \
                                                    n_tiles, kv, n_heads, n_kv_heads, q_stride, \
                                                    dense_keys, stream)
  if (head_dim == 64) {
    int8 ? KVB_LAUNCH(64, true) : KVB_LAUNCH(64, false);
  } else {
    int8 ? KVB_LAUNCH(128, true) : KVB_LAUNCH(128, false);
  }
#undef KVB_LAUNCH
  KVB_CUDA(cudaGetLastError());
}

}  // namespace

void kv_write(const f16* k, const f16* v, const StepRow* rows, const KvView& kv, int n_tokens,
              int n_kv_heads, int head_dim, int in_stride, cudaStream_t stream) {
  if (in_stride < n_kv_heads * head_dim) throw Error("kv_write rows overlap");
  if (kv.dtype == DType::I8) {
    const std::size_t n_vectors = static_cast<std::size_t>(n_tokens) * n_kv_heads;
    const unsigned grid = static_cast<unsigned>((n_vectors + kBlockWarps - 1) / kBlockWarps);
    kv_write_int8_kernel<<<grid, kBlockThreads, 0, stream>>>(
        as_half(k), as_half(v), rows, kv, n_kv_heads, head_dim, in_stride, n_vectors);
  } else {
    const std::size_t n = static_cast<std::size_t>(n_tokens) * n_kv_heads * head_dim;
    kv_write_kernel<<<blocks_for(n), kBlockThreads, 0, stream>>>(
        as_half(k), as_half(v), rows, kv, n_kv_heads, head_dim, in_stride, n);
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
               int n_heads, int n_kv_heads, int head_dim, int max_ctx, int fixed_ctx,
               KeyLoads loads, cudaStream_t stream) {
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
  if (fixed_ctx > max_ctx) throw Error("attention over a fixed context longer than max_ctx");
  if (fixed_ctx > kv.blocks_per_seq * kv.block_size) {
    throw Error("attention over a fixed context longer than a sequence's cache");
  }
  const std::size_t shared_bytes = attention_shared_bytes(head_dim, threads, max_ctx);
  const dim3 grid(n_heads, n_tokens);
  const int n_rep = n_heads / n_kv_heads;
  const bool wide = loads == KeyLoads::Wide;
#define KVB_LAUNCH(wide_keys, int8)                                           \
  attention_kernel<wide_keys, int8><<<grid, threads, shared_bytes, stream>>>( \
      as_half(q), as_half(out), rows, kv, n_heads, n_rep, head_dim, fixed_ctx)
  if (kv.dtype == DType::I8) {
    wide ? KVB_LAUNCH(true, true) : KVB_LAUNCH(false, true);
  } else {
    wide ? KVB_LAUNCH(true, false) : KVB_LAUNCH(false, false);
  }
#undef KVB_LAUNCH
  KVB_CUDA(cudaGetLastError());
}

bool attention_prefill_supports(const KvView& kv, int head_dim) {
  return kv.dim_stride == 1 && (head_dim == 64 || head_dim == 128) &&
         (kv.dtype == DType::F16 || kv.dtype == DType::I8);
}

void attention_prefill(const f16* q, f16* out, const StepRow* rows, const AttnTile* tiles,
                       int n_tiles, const KvView& kv, int n_heads, int n_kv_heads, int head_dim,
                       cudaStream_t stream) {
  if (!attention_prefill_supports(kv, head_dim)) {
    throw Error("prefill attention needs adjacent key dimensions and head_dim 64 or 128");
  }
  if (n_heads % n_kv_heads != 0) throw Error("query heads must be a multiple of KV heads");
  if (n_tiles == 0) return;
  dispatch_attention_prefill<true>(q, out, rows, tiles, n_tiles, kv, n_heads, n_kv_heads, head_dim,
                                   n_heads * head_dim, 0, stream);
}

void attention_dense(const f16* q, f16* out, const StepRow* rows, const AttnTile* tiles,
                     int n_tiles, const KvView& kv, int q_stride, int n_keys, int n_heads,
                     int n_kv_heads, int head_dim, cudaStream_t stream) {
  if (!attention_prefill_supports(kv, head_dim)) {
    throw Error("dense attention needs adjacent key dimensions and head_dim 64 or 128");
  }
  if (n_heads % n_kv_heads != 0) throw Error("query heads must be a multiple of KV heads");
  if (n_keys < 1 || n_keys > kv.blocks_per_seq * kv.block_size) {
    throw Error("dense attention needs between one key and as many as a sequence's cache holds");
  }
  // A query row is read 16 bytes at a time.
  if (q_stride < n_heads * head_dim || q_stride % 8 != 0) {
    throw Error("dense attention needs q rows of n_heads * head_dim or more, 16-byte aligned");
  }
  if (n_tiles == 0) return;
  dispatch_attention_prefill<false>(q, out, rows, tiles, n_tiles, kv, n_heads, n_kv_heads, head_dim,
                                    q_stride, n_keys, stream);
}

}  // namespace kvb::kernels
