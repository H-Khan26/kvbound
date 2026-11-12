#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "kvbound/kv_layout.h"
#include "kvbound/step.h"

// The engine's CUDA kernels. Each function launches one kernel on `stream` and returns without
// waiting for it. Matrices are row-major with one row per token; FP16 data is passed as raw
// 16-bit words so host code needs no half type.
namespace kvb::kernels {

using f16 = std::uint16_t;

// out[t] = table[rows[t].token_id], with `dim` columns.
void embedding(const f16* table, const StepRow* rows, f16* out, int n_tokens, int dim,
               cudaStream_t stream);

// out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * weight, accumulated in FP32.
void rms_norm(const f16* x, const f16* weight, f16* out, int n_rows, int dim, float eps,
              cudaStream_t stream);

// Rotates each (j, j + head_dim/2) pair of x [n_tokens, n_heads, head_dim] in place by the angle
// of its row's position. cos_sin holds {cos, sin} pairs, [max positions][head_dim/2].
void rope(f16* x, const StepRow* rows, const float* cos_sin, int n_tokens, int n_heads,
          int head_dim, cudaStream_t stream);

// gate[i] = silu(gate[i]) * up[i]
void silu_mul(f16* gate, const f16* up, std::size_t n, cudaStream_t stream);

// x[i] += y[i]
void add(f16* x, const f16* y, std::size_t n, cudaStream_t stream);

// out[o] = x[out_rows[o]], with `dim` columns.
void gather_rows(const f16* x, const std::int32_t* out_rows, f16* out, int n_out, int dim,
                 cudaStream_t stream);

// out[r] = index of the largest element of logits[r]; the lowest index wins ties.
void argmax(const float* logits, std::int32_t* out, int n_rows, int n_cols, cudaStream_t stream);

// Stores k and v [n_tokens, n_kv_heads, head_dim] at each row's kv_slot.
void kv_write(const f16* k, const f16* v, const StepRow* rows, const KvView& kv, int n_tokens,
              int n_kv_heads, int head_dim, cudaStream_t stream);

// How attention() reads a cached key when it computes scores.
enum class KeyLoads : std::uint8_t {
  Scalar,  // one FP16 element per load: works with every layout
  Wide,    // 16 bytes per load: needs adjacent dimensions (dim_stride 1) and head_dim % 8 == 0
};

std::string_view key_loads_name(KeyLoads loads) noexcept;
// Wide when the layout and head_dim allow it, else Scalar.
KeyLoads widest_key_loads(const KvView& kv, int head_dim) noexcept;

// Causal attention of each row's queries q [n_tokens, n_heads, head_dim] over the cached keys
// and values of its sequence, positions 0..row.position. Query head h reads KV head
// h / (n_heads / n_kv_heads). Scores and the softmax are FP32. max_ctx bounds row.position + 1.
void attention(const f16* q, f16* out, const StepRow* rows, const KvView& kv, int n_tokens,
               int n_heads, int n_kv_heads, int head_dim, int max_ctx, KeyLoads loads,
               cudaStream_t stream);

// Largest max_ctx attention() accepts for this head_dim (its scores live in shared memory);
// 0 when head_dim itself is unsupported.
int attention_max_positions(int head_dim);

// Weight conversion to FP16 at load time, rounding to nearest even.
void bf16_to_f16(f16* data, std::size_t n, cudaStream_t stream);  // in place
void f32_to_f16(const float* src, f16* dst, std::size_t n, cudaStream_t stream);
// Number of infinities and NaNs in data; waits for the result.
std::size_t count_non_finite(const f16* data, std::size_t n, cudaStream_t stream);

}  // namespace kvb::kernels
