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

// --- Tensor-core product ----------------------------------------------------------------------
//
// Each block computes a tile of kBM rows by kBN output channels of y. Stages of kBK inputs of x
// (FP16) and of the block's weight rows (packed integers) arrive in shared memory by cp.async,
// kStages - 1 ahead of the stage being multiplied, so DRAM streams the weights while the tensor
// cores work. Warps expand the integers to FP16 in registers right before each mma.sync, so
// weights cross DRAM and shared memory at their packed size and are never stored as FP16.
//
// The MMA computes y^T = W x^T: the expanded weights are its A operand (16 channels by 16
// inputs), which the expansion writes straight into the four registers A needs, and x is its B
// operand (16 inputs by 8 rows), whose two registers come from one row of x. Rows are padded to
// a multiple of 8.
//
// Within each 64-input chunk, lane group t (lane % 4) owns inputs 16t..16t+15 of both x and W,
// and MMA step s (of 4) uses inputs 16t+4s..16t+4s+3 as the fragment's k = {2t, 2t+1, 2t+8,
// 2t+9}. A dot product does not depend on the order of its terms, so x and W only need the same
// order. With it, one 16-byte read of a weight row (8 bytes for INT4) feeds four MMA steps and
// one 16-byte read of x feeds two.

// Bytes of one 64-input chunk of a weight row.
template <bool kInt4>
inline constexpr int kChunkBytes = kInt4 ? 32 : 64;

// Tile shape: rows and channels of y per block, inputs per stage, the warp grid (warps along
// rows and channels split the tile; warps along K take alternate 64-input chunks of each stage
// and add their sums at the end), and the stages in flight.
template <int BM, int BN, int BK, int WarpsM, int WarpsN, int WarpsK, int Stages>
struct GemmTile {
  static constexpr int kBM = BM, kBN = BN, kBK = BK, kStages = Stages;
  static constexpr int kWarpsM = WarpsM, kWarpsN = WarpsN, kWarpsK = WarpsK;
  static constexpr int kThreads = WarpsM * WarpsN * WarpsK * kWarpSize;
  static constexpr int kWM = BM / WarpsM, kWN = BN / WarpsN;  // rows and channels per warp
  static constexpr int kRowTiles = kWM / 8;                   // n8 tiles of rows per warp
  static constexpr int kChTiles = kWN / 16;                   // m16 tiles of channels per warp
  static constexpr int kChunks = BK / 64;
  // A row of x in shared memory is padded by 16 bytes, so the rows g and g + 1 that one
  // quarter-warp reads fall in different banks.
  static constexpr int kXRow = 2 * BK + 16;
  static constexpr int kXBytes = BM * kXRow;
  template <bool kInt4>
  static constexpr int kStageBytes = kXBytes + kChunks * BN * kChunkBytes<kInt4>;
  template <bool kInt4>
  static constexpr int kSmemBytes = Stages * kStageBytes<kInt4>;

  static_assert(kWM % 8 == 0 && kWN % 16 == 0 && kChunks % WarpsK == 0);
};

__device__ inline void cp_async16(void* smem, const void* global, bool valid) {
  // A source size of 0 fills the 16 bytes with zeros without reading global memory.
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(global),
               "r"(valid ? 16 : 0)
               : "memory");
}

__device__ inline void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

// Waits until at most N committed groups are still in flight.
template <int N>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// d += a * b for one m16n8k16 tile: FP16 inputs, FP32 sums.
__device__ inline void mma_f16(float* d, const unsigned* a, const unsigned* b) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline unsigned half2_bits(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }
__device__ inline __half2 bits_half2(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

// The four INT8 weights of `word` times `scale` as FP16 pairs {q0, q1} and {q2, q3}. Each byte,
// offset to unsigned, becomes the mantissa of an FP16 with exponent 2^10, which is 1024 + byte
// exactly; subtracting 1152 leaves q. q * scale is exact in FP32, so the FP16 product rounds
// it the same way as __float2half(q * scale), the value dequantize() writes.
__device__ inline void dequant_int8(unsigned word, __half2 scale, unsigned& lo, unsigned& hi) {
  const unsigned u = word ^ 0x80808080u;
  const __half2 bias = __float2half2_rn(1152.0f);
  lo = half2_bits(__hmul2(__hsub2(bits_half2(__byte_perm(u, 0x64646464u, 0x4140)), bias), scale));
  hi = half2_bits(__hmul2(__hsub2(bits_half2(__byte_perm(u, 0x64646464u, 0x4342)), bias), scale));
}

// The INT4 weights in byte `byte` of `word` (low nibble first) times `scale`, as in
// dequant_int8. The low nibble lands in the mantissa of the low FP16 (1024 + n) and the high
// nibble four bits up in the high FP16 (1024 + 16 n), so one multiply-add with {1, 1/16} turns
// both into integers; flipping bit 3 first makes the nibbles unsigned.
__device__ inline unsigned dequant_int4(unsigned word, int byte, __half2 scale) {
  const __half2 mul = __halves2half2(__float2half(1.0f), __float2half(1.0f / 16.0f));
  const __half2 add = __halves2half2(__float2half(-1032.0f), __float2half(-72.0f));
  const unsigned spread = __byte_perm(word, 0, byte * 0x1111);
  const unsigned h = (spread & 0x00F0000Fu) ^ 0x64806408u;
  return half2_bits(__hmul2(__hfma2(bits_half2(h), mul, add), scale));
}

// y [n_rows, n_out] = x [n_rows, n_in] * W^T on tensor cores. Blocks along x of the grid take
// successive row tiles of the same channels, so the blocks running at once share weights and
// each weight crosses the memory bus about once per launch rather than once per row tile.
// `group_shift` is log2 of the INT4 group size.
template <typename Tile, bool kInt4, bool kF32Out>
__global__ void __launch_bounds__(Tile::kThreads)
    gemm_quantized_kernel(const std::uint8_t* __restrict__ weight, const __half* __restrict__ scale,
                          const __half* __restrict__ x, void* __restrict__ y, int n_rows, int n_in,
                          int n_out, int group_shift) {
  constexpr int kBM = Tile::kBM, kBN = Tile::kBN, kBK = Tile::kBK, kThreads = Tile::kThreads;
  constexpr int kRowTiles = Tile::kRowTiles, kChTiles = Tile::kChTiles;
  constexpr int kXRow = Tile::kXRow, kXBytes = Tile::kXBytes;
  constexpr int kStageBytes = Tile::template kStageBytes<kInt4>;
  constexpr int kRowChunk = kChunkBytes<kInt4>;
  constexpr int kXPieces = kBK / 8;                  // 16-byte pieces of an x row per stage
  constexpr int kWPieces = kBK / (kInt4 ? 32 : 16);  // and of a weight row
  constexpr int kXCopies = (kBM * kXPieces + kThreads - 1) / kThreads;
  constexpr int kWCopies = (kBN * kWPieces + kThreads - 1) / kThreads;
  constexpr int kNever = 1 << 30;  // input offset of a copy that is always out of range

  extern __shared__ uint4 smem_words[];
  std::uint8_t* smem = reinterpret_cast<std::uint8_t*>(smem_words);

  const int lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
  const int g = lane / 4, t = lane % 4;
  const int wk = warp / (Tile::kWarpsM * Tile::kWarpsN);
  const int wmn = warp % (Tile::kWarpsM * Tile::kWarpsN);
  const int wm = wmn / Tile::kWarpsN, wn = wmn % Tile::kWarpsN;
  const int m0 = blockIdx.x * kBM, n0 = blockIdx.y * kBN;
  const int row_bytes = kInt4 ? n_in / 2 : n_in;

  // This thread's copies, the same for every stage but for the stage's first input. A copy
  // outside the matrix gets offset kNever, which fails the range test of every stage.
  const __half* x_src[kXCopies];
  int x_dst[kXCopies], x_off[kXCopies];
  for (int i = 0; i < kXCopies; ++i) {
    const int p = threadIdx.x + i * kThreads;
    const int r = p / kXPieces, q = p % kXPieces;
    const bool inside = p < kBM * kXPieces && m0 + r < n_rows;
    x_src[i] = inside ? x + static_cast<std::size_t>(m0 + r) * n_in + 8 * q : x;
    x_dst[i] = r * kXRow + 16 * q;
    x_off[i] = inside ? 8 * q : kNever;
  }
  const std::uint8_t* w_src[kWCopies];
  int w_dst[kWCopies], w_off[kWCopies];
  for (int i = 0; i < kWCopies; ++i) {
    constexpr int kPerChunk = kRowChunk / 16;
    const int p = threadIdx.x + i * kThreads;
    const int n = p / kWPieces, q = p % kWPieces;
    const bool inside = p < kBN * kWPieces && n0 + n < n_out;
    w_src[i] = inside ? weight + static_cast<std::size_t>(n0 + n) * row_bytes + 16 * q : weight;
    w_dst[i] = kXBytes + (q / kPerChunk) * kBN * kRowChunk + n * kRowChunk + 16 * (q % kPerChunk);
    w_off[i] = inside ? 16 * q : kNever;
  }
  const auto load_stage = [&](int stage, int kc) {
    std::uint8_t* base = smem + stage * kStageBytes;
    const int k0 = kc * kBK, b0 = kInt4 ? k0 / 2 : k0;
    for (int i = 0; i < kXCopies; ++i) {
      if (kBM * kXPieces % kThreads != 0 && threadIdx.x + i * kThreads >= kBM * kXPieces) break;
      const bool valid = x_off[i] < n_in - k0;
      cp_async16(base + x_dst[i], valid ? x_src[i] + k0 : x, valid);
    }
    for (int i = 0; i < kWCopies; ++i) {
      if (kBN * kWPieces % kThreads != 0 && threadIdx.x + i * kThreads >= kBN * kWPieces) break;
      const bool valid = w_off[i] < row_bytes - b0;
      cp_async16(base + w_dst[i], valid ? w_src[i] + b0 : weight, valid);
    }
  };

  // Weight rows g and g + 8 of each m16 tile of this lane; rows past n_out repeat the last one.
  int channels[kChTiles][2];
  for (int i = 0; i < kChTiles; ++i) {
    for (int h = 0; h < 2; ++h) {
      channels[i][h] = min(n0 + wn * Tile::kWN + 16 * i + g + 8 * h, n_out - 1);
    }
  }
  const int n_groups = n_in >> group_shift;
  __half2 scales[kChTiles][2];
  int scale_group = -1;
  if constexpr (!kInt4) {
    for (int i = 0; i < kChTiles; ++i) {
      for (int h = 0; h < 2; ++h) scales[i][h] = __half2half2(scale[channels[i][h]]);
    }
  }

  float acc[kChTiles][kRowTiles][4] = {};
  const int n_stages = (n_in + kBK - 1) / kBK;
  for (int s = 0; s < Tile::kStages - 1; ++s) {
    if (s < n_stages) load_stage(s, s);
    cp_async_commit();
  }
  for (int kc = 0; kc < n_stages; ++kc) {
    cp_async_wait<Tile::kStages - 2>();
    // Stage kc has landed for every thread, and every warp is done with the stage reloaded next.
    __syncthreads();
    if (kc + Tile::kStages - 1 < n_stages) {
      load_stage((kc + Tile::kStages - 1) % Tile::kStages, kc + Tile::kStages - 1);
    }
    cp_async_commit();

    const std::uint8_t* xs = smem + (kc % Tile::kStages) * kStageBytes;
    const std::uint8_t* ws = xs + kXBytes;
    for (int c = wk; c < Tile::kChunks; c += Tile::kWarpsK) {
      const int k = kc * kBK + 64 * c;
      if (k >= n_in) break;
      if constexpr (kInt4) {
        // A stage past the end of the row reads zeros; its scale only has to exist.
        const int group = min((k + 16 * t) >> group_shift, n_groups - 1);
        if (group != scale_group) {
          scale_group = group;
          for (int i = 0; i < kChTiles; ++i) {
            for (int h = 0; h < 2; ++h) {
              scales[i][h] =
                  __half2half2(scale[static_cast<std::size_t>(channels[i][h]) * n_groups + group]);
            }
          }
        }
      }
      // Packed weights of this lane for the chunk: 16 inputs of rows g and g + 8 per m16 tile.
      unsigned packed[kChTiles][2][kInt4 ? 2 : 4];
      for (int i = 0; i < kChTiles; ++i) {
        for (int h = 0; h < 2; ++h) {
          const std::uint8_t* src =
              ws + c * kBN * kRowChunk + (wn * Tile::kWN + 16 * i + g + 8 * h) * kRowChunk;
          if constexpr (kInt4) {
            const uint2 v = *reinterpret_cast<const uint2*>(src + 8 * t);
            packed[i][h][0] = v.x;
            packed[i][h][1] = v.y;
          } else {
            const uint4 v = *reinterpret_cast<const uint4*>(src + 16 * t);
            packed[i][h][0] = v.x;
            packed[i][h][1] = v.y;
            packed[i][h][2] = v.z;
            packed[i][h][3] = v.w;
          }
        }
      }
      for (int half = 0; half < 2; ++half) {
        // x for MMA steps 2 half and 2 half + 1: row g of each n8 tile.
        uint4 xb[kRowTiles];
        for (int j = 0; j < kRowTiles; ++j) {
          const int r = wm * Tile::kWM + 8 * j + g;
          xb[j] = *reinterpret_cast<const uint4*>(xs + r * kXRow + 128 * c + 32 * t + 16 * half);
        }
        for (int step = 0; step < 2; ++step) {
          for (int i = 0; i < kChTiles; ++i) {
            // A registers: {row g, k 0-1}, {row g + 8, k 0-1}, {row g, k 2-3}, {row g + 8, k 2-3}.
            unsigned a[4];
            for (int h = 0; h < 2; ++h) {
              if constexpr (kInt4) {
                // Step 2 half + step reads bytes 4 half + 2 step and the next of the lane's 8.
                a[h] = dequant_int4(packed[i][h][half], 2 * step, scales[i][h]);
                a[h + 2] = dequant_int4(packed[i][h][half], 2 * step + 1, scales[i][h]);
              } else {
                dequant_int8(packed[i][h][2 * half + step], scales[i][h], a[h], a[h + 2]);
              }
            }
            for (int j = 0; j < kRowTiles; ++j) {
              const unsigned b[2] = {step == 0 ? xb[j].x : xb[j].z, step == 0 ? xb[j].y : xb[j].w};
              mma_f16(acc[i][j], a, b);
            }
          }
        }
      }
    }
  }

  if constexpr (Tile::kWarpsK > 1) {
    // Warps past the first along K hand their sums to it through the now idle stage buffers.
    constexpr int kAcc = kChTiles * kRowTiles * 4;
    constexpr int kWarpsMN = Tile::kWarpsM * Tile::kWarpsN;
    static_assert((Tile::kWarpsK - 1) * kWarpsMN * kAcc * kWarpSize * 4 <=
                  Tile::template kSmemBytes<kInt4>);
    cp_async_wait<0>();
    __syncthreads();
    float* partial = reinterpret_cast<float*>(smem);
    float* flat = &acc[0][0][0];
    if (wk > 0) {
      for (int e = 0; e < kAcc; ++e) {
        partial[(((wk - 1) * kWarpsMN + wmn) * kAcc + e) * kWarpSize + lane] = flat[e];
      }
    }
    __syncthreads();
    if (wk > 0) return;
    for (int other = 0; other < Tile::kWarpsK - 1; ++other) {
      for (int e = 0; e < kAcc; ++e) {
        flat[e] += partial[((other * kWarpsMN + wmn) * kAcc + e) * kWarpSize + lane];
      }
    }
  }

  // Sums e of a tile: channel g + 8 (e / 2), row 2t + e % 2.
  for (int i = 0; i < kChTiles; ++i) {
    for (int j = 0; j < kRowTiles; ++j) {
      for (int e = 0; e < 4; ++e) {
        const int channel = n0 + wn * Tile::kWN + 16 * i + g + 8 * (e / 2);
        const int row = m0 + wm * Tile::kWM + 8 * j + 2 * t + e % 2;
        if (channel >= n_out || row >= n_rows) continue;
        const std::size_t at = static_cast<std::size_t>(row) * n_out + channel;
        if constexpr (kF32Out) {
          static_cast<float*>(y)[at] = acc[i][j][e];
        } else {
          static_cast<__half*>(y)[at] = __float2half(acc[i][j][e]);
        }
      }
    }
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

template <typename Tile, bool kInt4, bool kF32Out>
void launch_gemm(const WeightRef& w, const f16* x, void* y, int n_rows, int n_in, int n_out,
                 cudaStream_t stream) {
  const auto kernel = gemm_quantized_kernel<Tile, kInt4, kF32Out>;
  constexpr int kSmem = Tile::template kSmemBytes<kInt4>;
  // Dynamic shared memory above 48 KiB needs an opt-in, once per kernel.
  static const bool opted_in = [&] {
    KVB_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    return true;
  }();
  (void)opted_in;
  const dim3 grid(static_cast<unsigned>((n_rows + Tile::kBM - 1) / Tile::kBM),
                  static_cast<unsigned>((n_out + Tile::kBN - 1) / Tile::kBN));
  const int group_shift = kInt4 ? std::countr_zero(static_cast<unsigned>(w.group_size)) : 0;
  kernel<<<grid, Tile::kThreads, kSmem, stream>>>(static_cast<const std::uint8_t*>(w.data),
                                                  as_half(w.scale), as_half(x), y, n_rows, n_in,
                                                  n_out, group_shift);
  KVB_CUDA(cudaGetLastError());
}

// Tiles measured on an RTX 4070. Up to about 64 rows the product streams weights from DRAM:
// small row tiles with deep stages split across warps along K keep the most weight bytes in
// flight. Wider steps need tensor-core throughput: 64 x 128 tiles of four warps, small enough
// in shared memory for two blocks per SM, whenever they fill the GPU twice over.
using DeepTile32 = GemmTile<32, 64, 256, 1, 2, 4, 2>;
using DeepTile48 = GemmTile<48, 64, 128, 1, 2, 2, 4>;
using DeepTile64 = GemmTile<64, 64, 128, 2, 2, 2, 3>;
using WideTile = GemmTile<64, 128, 64, 1, 4, 1, 2>;

int sm_count() {
  static const int count = [] {
    int device = 0, n = 0;
    KVB_CUDA(cudaGetDevice(&device));
    KVB_CUDA(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device));
    return n;
  }();
  return count;
}

template <bool kInt4, bool kF32Out>
void dispatch_gemm(const WeightRef& w, const f16* x, void* y, int n_rows, int n_in, int n_out,
                   cudaStream_t stream) {
  const auto blocks = [&](int rows, int channels) {
    return static_cast<long>((n_rows + rows - 1) / rows) * ((n_out + channels - 1) / channels);
  };
#define KVB_LAUNCH(tile) launch_gemm<tile, kInt4, kF32Out>(w, x, y, n_rows, n_in, n_out, stream)
  if (n_rows > 96 && blocks(WideTile::kBM, WideTile::kBN) >= 2 * sm_count()) {
    KVB_LAUNCH(WideTile);
  } else if (n_rows <= 32 || (kInt4 && n_rows > 48) || n_rows > 96) {
    KVB_LAUNCH(DeepTile32);
  } else if (n_rows <= 48 || n_rows > 64) {
    KVB_LAUNCH(DeepTile48);
  } else {
    KVB_LAUNCH(DeepTile64);
  }
#undef KVB_LAUNCH
}

}  // namespace

void gemm_quantized(const WeightRef& w, const f16* x, void* y, int n_rows, int n_in, int n_out,
                    bool f32_out, cudaStream_t stream) {
  check_quantized(w, n_in);
  // cp.async copies 16 bytes at a time.
  if ((reinterpret_cast<std::uintptr_t>(x) | reinterpret_cast<std::uintptr_t>(w.data)) % 16 != 0) {
    throw Error("gemm_quantized needs x and the weights 16-byte aligned");
  }
  if (n_rows <= 0 || n_out <= 0) return;
  const bool int4 = w.scheme == WeightScheme::Int4Grouped;
  if (int4 && f32_out) dispatch_gemm<true, true>(w, x, y, n_rows, n_in, n_out, stream);
  if (int4 && !f32_out) dispatch_gemm<true, false>(w, x, y, n_rows, n_in, n_out, stream);
  if (!int4 && f32_out) dispatch_gemm<false, true>(w, x, y, n_rows, n_in, n_out, stream);
  if (!int4 && !f32_out) dispatch_gemm<false, false>(w, x, y, n_rows, n_in, n_out, stream);
}

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
