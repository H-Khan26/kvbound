// Python bindings: each kernel as a function over NumPy arrays, the Engine and the AsrEngine.
// Arrays cross the boundary on the host; FP16 data travels as uint16 (view it with
// .view(np.float16)).
// Meant for tests and evals, where a copy per call does not matter.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "kvbound/cuda/asr_engine.h"
#include "kvbound/cuda/blas.h"
#include "kvbound/cuda/engine.h"
#include "kvbound/cuda/generate.h"
#include "kvbound/cuda/kernels.h"
#include "kvbound/cuda/linear.h"
#include "kvbound/cuda/serve.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using kvb::StepRow;
using kvb::cuda::DeviceBuffer;
using kvb::kernels::f16;

template <typename T>
using In = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;
template <typename T>
using Out = nb::ndarray<nb::numpy, T, nb::c_contig>;

template <typename T>
DeviceBuffer upload(const In<T>& array) {
  DeviceBuffer buffer(array.nbytes());
  if (array.nbytes() == 0) return buffer;
  KVB_CUDA(cudaMemcpy(buffer.as(), array.data(), array.nbytes(), cudaMemcpyHostToDevice));
  return buffer;
}

// Waits for the default stream and copies a device buffer into a new NumPy array.
template <typename T>
Out<T> download(const void* device, std::initializer_list<std::size_t> shape) {
  std::size_t n = 1;
  for (const std::size_t extent : shape) n *= extent;
  T* host = new T[n];
  nb::capsule owner(host, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  KVB_CUDA(cudaMemcpy(host, device, n * sizeof(T), cudaMemcpyDeviceToHost));
  return Out<T>(host, shape, owner);
}

template <typename T>
Out<T> to_numpy(const std::vector<T>& values, std::initializer_list<std::size_t> shape) {
  T* host = new T[values.size()];
  std::copy(values.begin(), values.end(), host);
  nb::capsule owner(host, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return Out<T>(host, shape, owner);
}

// Rows for kernels that only read token ids or positions.
DeviceBuffer upload_rows(const std::vector<StepRow>& rows) {
  DeviceBuffer buffer(rows.size() * sizeof(StepRow));
  KVB_CUDA(cudaMemcpy(buffer.as(), rows.data(), buffer.bytes(), cudaMemcpyHostToDevice));
  return buffer;
}

// One row per position, for kernels that read nothing else of a row.
DeviceBuffer upload_positions(const In<std::int32_t>& positions) {
  std::vector<StepRow> rows;
  for (std::size_t i = 0; i < positions.shape(0); ++i) {
    rows.push_back({.position = positions.data()[i]});
  }
  return upload_rows(rows);
}

// Tiles of the rows for attention_prefill() and attention_dense(), on the device.
struct DeviceTiles {
  DeviceBuffer data;
  int count;

  explicit DeviceTiles(std::span<const StepRow> rows) {
    const std::vector<kvb::AttnTile> tiles =
        kvb::plan_attention_tiles(rows, kvb::kernels::kAttnTileRows);
    count = static_cast<int>(tiles.size());
    data = DeviceBuffer(std::max<std::size_t>(tiles.size(), 1) * sizeof(kvb::AttnTile));
    KVB_CUDA(cudaMemcpy(data.as(), tiles.data(), tiles.size() * sizeof(kvb::AttnTile),
                        cudaMemcpyHostToDevice));
  }
};

// An optional bias vector on the device; nullptr when there is none.
struct DeviceBias {
  DeviceBuffer data;

  explicit DeviceBias(const std::optional<In<f16>>& bias) {
    if (bias) data = upload(*bias);
  }
  const f16* get() const { return data.bytes() > 0 ? data.as<f16>() : nullptr; }
};

kvb::KvLayout layout_from(const std::string& name) {
  if (const auto layout = kvb::kv_layout_from_name(name)) return *layout;
  throw kvb::Error("unknown KV layout " + name);
}

kvb::DType kv_dtype_from(const std::string& name) {
  if (const auto dtype = kvb::kv_dtype_from_name(name)) return *dtype;
  throw kvb::Error("unknown KV dtype " + name);
}

// A quantized matrix on the device: `data` holds int8 values or packed 4-bit pairs.
struct DeviceWeight {
  DeviceBuffer data, scale;
  kvb::WeightRef ref;

  DeviceWeight(const std::string& scheme, const In<std::uint8_t>& host_data,
               const In<f16>& host_scale, int group_size)
      : data(upload(host_data)), scale(upload(host_scale)) {
    const auto parsed = kvb::weight_scheme_from_name(scheme);
    if (!parsed || *parsed == kvb::WeightScheme::F16) {
      throw kvb::Error("not a quantized scheme: " + scheme);
    }
    ref = {
        .scheme = *parsed, .data = data.as(), .scale = scale.as<f16>(), .group_size = group_size};
  }
  // Shape [n_out, n_in] of the matrix a stored [n_out, bytes per row] tensor stands for.
  int n_in(const In<std::uint8_t>& host_data) const {
    const int stored = static_cast<int>(host_data.shape(1));
    return ref.scheme == kvb::WeightScheme::Int4Grouped ? 2 * stored : stored;
  }
};

kvb::cuda::GemmAccumulate accumulate_from(const std::string& name) {
  if (const auto mode = kvb::cuda::gemm_accumulate_from_name(name)) return *mode;
  throw kvb::Error("unknown GEMM accumulate mode " + name);
}

// --- Kernels
// --------------------------------------------------------------------------------------

Out<f16> rms_norm(In<f16> x, In<f16> weight, float eps) {
  const int rows = static_cast<int>(x.shape(0)), dim = static_cast<int>(x.shape(1));
  const DeviceBuffer dx = upload(x), dw = upload(weight), out(x.nbytes());
  kvb::kernels::rms_norm(dx.as<f16>(), dw.as<f16>(), out.as<f16>(), rows, dim, eps, nullptr);
  return download<f16>(out.as(), {x.shape(0), x.shape(1)});
}

Out<f16> layer_norm(In<f16> x, In<f16> weight, In<f16> bias, float eps) {
  const int rows = static_cast<int>(x.shape(0)), dim = static_cast<int>(x.shape(1));
  const DeviceBuffer dx = upload(x), dw = upload(weight), db = upload(bias), out(x.nbytes());
  kvb::kernels::layer_norm(dx.as<f16>(), dw.as<f16>(), db.as<f16>(), out.as<f16>(), rows, dim, eps,
                           nullptr);
  return download<f16>(out.as(), {x.shape(0), x.shape(1)});
}

// Returns {x + y, layer_norm(x + y)}, both [rows, dim].
std::vector<Out<f16>> add_layer_norm(In<f16> x, In<f16> y, In<f16> weight, In<f16> bias,
                                     float eps) {
  const int rows = static_cast<int>(x.shape(0)), dim = static_cast<int>(x.shape(1));
  const DeviceBuffer dx = upload(x), dy = upload(y), dw = upload(weight), db = upload(bias),
                     out(x.nbytes());
  kvb::kernels::add_layer_norm(dx.as<f16>(), dy.as<f16>(), dw.as<f16>(), db.as<f16>(),
                               out.as<f16>(), rows, dim, eps, nullptr);
  return {download<f16>(dx.as(), {x.shape(0), x.shape(1)}),
          download<f16>(out.as(), {x.shape(0), x.shape(1)})};
}

Out<f16> gelu(In<f16> x) {
  const DeviceBuffer dx = upload(x);
  kvb::kernels::gelu(dx.as<f16>(), dx.as<f16>(), x.size(), nullptr);
  return download<f16>(dx.as(), {x.size()});
}

// x [rows, dim] plus row positions[r] of table [n, dim].
Out<f16> add_positions(In<f16> x, In<f16> table, In<std::int32_t> positions) {
  const DeviceBuffer dx = upload(x), dt = upload(table), dr = upload_positions(positions),
                     out(x.nbytes());
  kvb::kernels::add_positions(dx.as<f16>(), dt.as<f16>(), dr.as<StepRow>(), out.as<f16>(),
                              static_cast<int>(x.shape(0)), static_cast<int>(x.shape(1)), nullptr);
  return download<f16>(out.as(), {x.shape(0), x.shape(1)});
}

// x [n_seqs * in_frames, channels], each sequence's frames in order. Returns
// [n_seqs * out_frames, 3 * channels].
Out<f16> unfold3(In<f16> x, int n_seqs, int stride) {
  const int in_frames = static_cast<int>(x.shape(0)) / n_seqs;
  const int channels = static_cast<int>(x.shape(1));
  const std::size_t out_rows = static_cast<std::size_t>(n_seqs) * ((in_frames - 1) / stride + 1);
  const DeviceBuffer dx = upload(x), out(out_rows * 3 * channels * sizeof(f16));
  kvb::kernels::unfold3(dx.as<f16>(), out.as<f16>(), n_seqs, in_frames, channels, stride, nullptr);
  return download<f16>(out.as(), {out_rows, 3 * x.shape(1)});
}

// y [rows, n_out] plus bias [n_out] in every row; y is float16 bits or float32.
Out<f16> add_bias(In<f16> y, In<f16> bias) {
  const DeviceBuffer dy = upload(y), db = upload(bias);
  kvb::kernels::add_bias(dy.as(), db.as<f16>(), static_cast<int>(y.shape(0)),
                         static_cast<int>(y.shape(1)), false, nullptr);
  return download<f16>(dy.as(), {y.shape(0), y.shape(1)});
}

Out<float> add_bias_f32(In<float> y, In<f16> bias) {
  const DeviceBuffer dy = upload(y), db = upload(bias);
  kvb::kernels::add_bias(dy.as(), db.as<f16>(), static_cast<int>(y.shape(0)),
                         static_cast<int>(y.shape(1)), true, nullptr);
  return download<float>(dy.as(), {y.shape(0), y.shape(1)});
}

// logits [n_out, vocab] of the step rows out_rows, a step's rows being at `positions`. Returns
// them with the tokens of `always` at -inf and, in rows at begin_pos, those of `begin`.
Out<float> suppress_logits(In<float> logits, In<std::int32_t> positions, In<std::int32_t> out_rows,
                           In<std::int32_t> always, In<std::int32_t> begin, int begin_pos) {
  const DeviceBuffer dl = upload(logits), dr = upload_positions(positions), dout = upload(out_rows),
                     da = upload(always), db = upload(begin);
  kvb::kernels::suppress_logits(
      dl.as<float>(), dr.as<StepRow>(), dout.as<std::int32_t>(), static_cast<int>(logits.shape(0)),
      static_cast<int>(logits.shape(1)), da.as<std::int32_t>(), static_cast<int>(always.shape(0)),
      db.as<std::int32_t>(), static_cast<int>(begin.shape(0)), begin_pos, nullptr);
  return download<float>(dl.as(), {logits.shape(0), logits.shape(1)});
}

Out<f16> embedding(In<f16> table, In<std::int32_t> ids) {
  std::vector<StepRow> rows;
  for (std::size_t i = 0; i < ids.shape(0); ++i) rows.push_back({.token_id = ids.data()[i]});
  const int dim = static_cast<int>(table.shape(1));
  const DeviceBuffer dt = upload(table), dr = upload_rows(rows),
                     out(rows.size() * dim * sizeof(f16));
  kvb::kernels::embedding(dt.as<f16>(), dr.as<StepRow>(), out.as<f16>(),
                          static_cast<int>(rows.size()), dim, nullptr);
  return download<f16>(out.as(), {rows.size(), table.shape(1)});
}

// x [tokens, heads, head_dim]; cos_sin [positions, head_dim / 2, 2].
Out<f16> rope(In<f16> x, In<std::int32_t> positions, In<float> cos_sin) {
  std::vector<StepRow> rows;
  for (std::size_t i = 0; i < positions.shape(0); ++i) {
    rows.push_back({.position = positions.data()[i]});
  }
  const DeviceBuffer dx = upload(x), dr = upload_rows(rows), table = upload(cos_sin);
  kvb::kernels::rope(dx.as<f16>(), dr.as<StepRow>(), table.as<float>(),
                     static_cast<int>(x.shape(0)), static_cast<int>(x.shape(1)),
                     static_cast<int>(x.shape(2)), nullptr);
  return download<f16>(dx.as(), {x.shape(0), x.shape(1), x.shape(2)});
}

Out<f16> silu_mul(In<f16> gate, In<f16> up) {
  const DeviceBuffer dg = upload(gate), du = upload(up);
  const int n = static_cast<int>(gate.size());
  kvb::kernels::silu_mul(dg.as<f16>(), du.as<f16>(), dg.as<f16>(), 1, n, n, nullptr);
  return download<f16>(dg.as(), {gate.size()});
}

// gate_up [rows, 2 * dim]: each row's gate half, then its up half, as the stacked product writes
// them. Returns [rows, dim].
Out<f16> silu_mul_stacked(In<f16> gate_up) {
  const int rows = static_cast<int>(gate_up.shape(0)), dim = static_cast<int>(gate_up.shape(1)) / 2;
  const DeviceBuffer dx = upload(gate_up), out(static_cast<std::size_t>(rows) * dim * sizeof(f16));
  kvb::kernels::silu_mul(dx.as<f16>(), dx.as<f16>() + dim, out.as<f16>(), rows, dim, 2 * dim,
                         nullptr);
  return download<f16>(out.as(), {gate_up.shape(0), static_cast<std::size_t>(dim)});
}

// Returns {x + y, rms_norm(x + y)}, both [rows, dim].
std::vector<Out<f16>> add_rms_norm(In<f16> x, In<f16> y, In<f16> weight, float eps) {
  const int rows = static_cast<int>(x.shape(0)), dim = static_cast<int>(x.shape(1));
  const DeviceBuffer dx = upload(x), dy = upload(y), dw = upload(weight), out(x.nbytes());
  kvb::kernels::add_rms_norm(dx.as<f16>(), dy.as<f16>(), dw.as<f16>(), out.as<f16>(), rows, dim,
                             eps, nullptr);
  return {download<f16>(dx.as(), {x.shape(0), x.shape(1)}),
          download<f16>(out.as(), {x.shape(0), x.shape(1)})};
}

Out<f16> add(In<f16> x, In<f16> y) {
  const DeviceBuffer dx = upload(x), dy = upload(y);
  kvb::kernels::add(dx.as<f16>(), dy.as<f16>(), x.size(), nullptr);
  return download<f16>(dx.as(), {x.size()});
}

Out<f16> gather_rows(In<f16> x, In<std::int32_t> out_rows) {
  const int dim = static_cast<int>(x.shape(1));
  const DeviceBuffer dx = upload(x), dr = upload(out_rows),
                     out(out_rows.shape(0) * dim * sizeof(f16));
  kvb::kernels::gather_rows(dx.as<f16>(), dr.as<std::int32_t>(), out.as<f16>(),
                            static_cast<int>(out_rows.shape(0)), dim, nullptr);
  return download<f16>(out.as(), {out_rows.shape(0), x.shape(1)});
}

Out<std::int32_t> argmax(In<float> logits) {
  const DeviceBuffer dl = upload(logits), out(logits.shape(0) * sizeof(std::int32_t));
  kvb::kernels::argmax(dl.as<float>(), out.as<std::int32_t>(), static_cast<int>(logits.shape(0)),
                       static_cast<int>(logits.shape(1)), nullptr);
  return download<std::int32_t>(out.as(), {logits.shape(0)});
}

// weight [n_out, n_in], x [rows, n_in]. Returns float16 bits, or float32 when f32_out.
nb::object linear(In<f16> weight, In<f16> x, const std::string& accumulate, bool f32_out) {
  const int n_out = static_cast<int>(weight.shape(0)), n_in = static_cast<int>(weight.shape(1));
  const int rows = static_cast<int>(x.shape(0));
  kvb::cuda::Blas blas(nullptr, accumulate_from(accumulate));
  const std::size_t n = static_cast<std::size_t>(rows) * n_out;
  const DeviceBuffer dw = upload(weight), dx = upload(x),
                     out(n * (f32_out ? sizeof(float) : sizeof(f16)));
  blas.linear(dw.as<f16>(), dx.as<f16>(), out.as(), rows, n_in, n_out,
              f32_out ? kvb::DType::F32 : kvb::DType::F16);
  if (f32_out) return nb::cast(download<float>(out.as(), {x.shape(0), weight.shape(0)}));
  return nb::cast(download<f16>(out.as(), {x.shape(0), weight.shape(0)}));
}

// weight [n_out, n_in], x [rows, n_in] through kvbound's FP16 kernel, plus bias [n_out] if
// given. Returns float16 bits, or float32 when f32_out.
nb::object linear_f16(In<f16> weight, In<f16> x, bool f32_out, std::optional<In<f16>> bias) {
  const int n_out = static_cast<int>(weight.shape(0)), n_in = static_cast<int>(weight.shape(1));
  const int rows = static_cast<int>(x.shape(0));
  const std::size_t n = static_cast<std::size_t>(rows) * n_out;
  const DeviceBuffer dw = upload(weight), dx = upload(x),
                     out(n * (f32_out ? sizeof(float) : sizeof(f16)));
  const DeviceBias db(bias);
  kvb::kernels::linear_f16(dw.as<f16>(), db.get(), dx.as<f16>(), out.as(), rows, n_in, n_out,
                           f32_out, nullptr);
  if (f32_out) return nb::cast(download<float>(out.as(), {x.shape(0), weight.shape(0)}));
  return nb::cast(download<f16>(out.as(), {x.shape(0), weight.shape(0)}));
}

// x [rows, n_in] times a quantized matrix stored as `data` [n_out, bytes per row] (int8 values
// or packed 4-bit pairs, as uint8) with FP16 `scale`, plus bias [n_out] if given. Returns
// float16 bits, or float32.
nb::object linear_quantized(const std::string& scheme, In<std::uint8_t> data, In<f16> scale,
                            In<f16> x, int group_size, bool f32_out, std::optional<In<f16>> bias) {
  const DeviceWeight w(scheme, data, scale, group_size);
  const int n_out = static_cast<int>(data.shape(0)), n_in = w.n_in(data);
  const int rows = static_cast<int>(x.shape(0));
  const std::size_t n = static_cast<std::size_t>(rows) * n_out;
  const DeviceBuffer dx = upload(x), out(n * (f32_out ? sizeof(float) : sizeof(f16)));
  const DeviceBias db(bias);
  kvb::kernels::linear_quantized(w.ref, db.get(), dx.as<f16>(), out.as(), rows, n_in, n_out,
                                 f32_out, nullptr);
  if (f32_out) return nb::cast(download<float>(out.as(), {x.shape(0), data.shape(0)}));
  return nb::cast(download<f16>(out.as(), {x.shape(0), data.shape(0)}));
}

// The same product as linear_quantized on tensor cores, for wide steps.
nb::object gemm_quantized(const std::string& scheme, In<std::uint8_t> data, In<f16> scale,
                          In<f16> x, int group_size, bool f32_out, std::optional<In<f16>> bias) {
  const DeviceWeight w(scheme, data, scale, group_size);
  const int n_out = static_cast<int>(data.shape(0)), n_in = w.n_in(data);
  const int rows = static_cast<int>(x.shape(0));
  const std::size_t n = static_cast<std::size_t>(rows) * n_out;
  const DeviceBuffer dx = upload(x), out(n * (f32_out ? sizeof(float) : sizeof(f16)));
  const DeviceBias db(bias);
  kvb::kernels::gemm_quantized(w.ref, db.get(), dx.as<f16>(), out.as(), rows, n_in, n_out, f32_out,
                               nullptr);
  if (f32_out) return nb::cast(download<float>(out.as(), {x.shape(0), data.shape(0)}));
  return nb::cast(download<f16>(out.as(), {x.shape(0), data.shape(0)}));
}

// x [rows, n_in] times a matrix plus bias [n_out] if given, through run_linear(): the kernel
// the engine picks for the scheme, the rows and the three options. A quantized matrix is passed
// as to linear_quantized; for scheme "fp16", `data` is the FP16 matrix [n_out, n_in] viewed as
// bytes, [n_out, 2 * n_in], and `scale` is not read. Returns float16 bits, or float32.
nb::object run_linear(const std::string& scheme, In<std::uint8_t> data, In<f16> scale, In<f16> x,
                      std::optional<In<f16>> bias, int group_size, bool f32_out, int f16_gemv_rows,
                      int quantized_gemm_rows, bool expand_quantized,
                      std::size_t dequant_slice_bytes, const std::string& accumulate) {
  const int n_out = static_cast<int>(data.shape(0)), rows = static_cast<int>(x.shape(0));
  DeviceBuffer f16_weight;
  std::optional<DeviceWeight> quantized;
  kvb::WeightRef ref;
  int n_in = static_cast<int>(data.shape(1)) / 2;
  if (scheme == "fp16") {
    f16_weight = upload(data);
    ref = {.scheme = kvb::WeightScheme::F16, .data = f16_weight.as()};
  } else {
    quantized.emplace(scheme, data, scale, group_size);
    ref = quantized->ref;
    n_in = quantized->n_in(data);
  }
  const DeviceBuffer dequant(quantized && expand_quantized
                                 ? kvb::cuda::expand_scratch_bytes(n_in, n_out, dequant_slice_bytes)
                                 : 0);
  kvb::cuda::Blas blas(nullptr, accumulate_from(accumulate));
  const std::size_t n = static_cast<std::size_t>(rows) * n_out;
  const DeviceBuffer dx = upload(x), out(n * (f32_out ? sizeof(float) : sizeof(f16)));
  const DeviceBias db(bias);
  kvb::cuda::run_linear(ref, db.get(), dx.as<f16>(), out.as(), rows, n_in, n_out,
                        f32_out ? kvb::DType::F32 : kvb::DType::F16,
                        {.f16_gemv_rows = f16_gemv_rows,
                         .quantized_gemm_rows = quantized_gemm_rows,
                         .expand_quantized = expand_quantized},
                        blas, dequant, nullptr);
  if (f32_out) return nb::cast(download<float>(out.as(), {x.shape(0), data.shape(0)}));
  return nb::cast(download<f16>(out.as(), {x.shape(0), data.shape(0)}));
}

// Output channels first_channel onward of the quantized matrix, expanded to FP16
// [n_channels, n_in]. n_channels 0 means every channel from first_channel.
Out<f16> dequantize(const std::string& scheme, In<std::uint8_t> data, In<f16> scale, int group_size,
                    int first_channel, int n_channels) {
  const DeviceWeight w(scheme, data, scale, group_size);
  const int n_out = static_cast<int>(data.shape(0)), n_in = w.n_in(data);
  const int count = n_channels > 0 ? n_channels : n_out - first_channel;
  if (first_channel < 0 || count <= 0 || first_channel + count > n_out) {
    throw kvb::Error("channel range outside the matrix");
  }
  const DeviceBuffer out(static_cast<std::size_t>(count) * n_in * sizeof(f16));
  kvb::kernels::dequantize(w.ref, out.as<f16>(), n_in, first_channel, count, nullptr);
  return download<f16>(out.as(), {static_cast<std::size_t>(count), static_cast<std::size_t>(n_in)});
}

// Rows `ids` of a quantized table, as FP16.
Out<f16> embedding_quantized(const std::string& scheme, In<std::uint8_t> data, In<f16> scale,
                             In<std::int32_t> ids, int group_size) {
  const DeviceWeight w(scheme, data, scale, group_size);
  const int dim = w.n_in(data);
  std::vector<StepRow> rows;
  for (std::size_t i = 0; i < ids.shape(0); ++i) rows.push_back({.token_id = ids.data()[i]});
  const DeviceBuffer dr = upload_rows(rows), out(rows.size() * dim * sizeof(f16));
  kvb::kernels::embedding_quantized(w.ref, dr.as<StepRow>(), out.as<f16>(),
                                    static_cast<int>(rows.size()), dim, nullptr);
  return download<f16>(out.as(), {rows.size(), static_cast<std::size_t>(dim)});
}

// Writes k and v [cached, kv_heads, head_dim] into a fresh cache at the kv_slot of the first
// `cached` tokens, then attends q [tokens, heads, head_dim] over it for the tokens from
// query_from on. block_table is [sequences, blocks_per_seq]. kernel "decode" runs attention();
// "prefill" plans tiles over the query rows and runs attention_prefill(). With fixed_ctx > 0
// every query attends positions 0..fixed_ctx - 1 of its sequence: "decode" passes it to
// attention(), and "dense" runs attention_dense() over tiles of the query rows. With stacked_kv
// the cache is written from one value that holds each token's keys, then its values.
Out<f16> kv_attention(In<f16> q, In<f16> k, In<f16> v, In<std::int32_t> positions,
                      In<std::int32_t> seqs, In<std::int32_t> kv_slots,
                      In<std::int32_t> block_table, int block_size, int n_blocks,
                      const std::string& layout, int query_from, bool wide_key_loads,
                      const std::string& kv_dtype, const std::string& kernel, int fixed_ctx,
                      bool stacked_kv) {
  if (kernel != "decode" && kernel != "prefill" && kernel != "dense") {
    throw kvb::Error("unknown kernel " + kernel);
  }
  if (kernel == "dense" ? fixed_ctx <= 0 : kernel == "prefill" && fixed_ctx > 0) {
    throw kvb::Error("dense attention needs a fixed_ctx and prefill attention takes none");
  }
  const int n_tokens = static_cast<int>(q.shape(0)), n_heads = static_cast<int>(q.shape(1));
  const int head_dim = static_cast<int>(q.shape(2)), n_kv_heads = static_cast<int>(k.shape(1));
  const int blocks_per_seq = static_cast<int>(block_table.shape(1));
  kvb::cuda::KvPool pool(kvb::KvDims{.n_layers = 1,
                                     .n_blocks = n_blocks,
                                     .n_kv_heads = n_kv_heads,
                                     .block_size = block_size,
                                     .head_dim = head_dim},
                         layout_from(layout), static_cast<int>(block_table.shape(0)),
                         blocks_per_seq, kv_dtype_from(kv_dtype));
  std::vector<StepRow> rows;
  int max_ctx = std::max(1, fixed_ctx);
  for (int i = 0; i < n_tokens; ++i) {
    rows.push_back({.token_id = 0,
                    .position = positions.data()[i],
                    .seq = seqs.data()[i],
                    .kv_slot = kv_slots.data()[i]});
    max_ctx = std::max(max_ctx, positions.data()[i] + 1);
  }
  for (std::size_t seq = 0; seq < block_table.shape(0); ++seq) {
    std::vector<int> blocks(block_table.data() + seq * blocks_per_seq,
                            block_table.data() + (seq + 1) * blocks_per_seq);
    pool.set_blocks(static_cast<int>(seq), blocks, nullptr);
  }
  const DeviceBuffer dq = upload(q), dk = upload(k), dv = upload(v), dr = upload_rows(rows),
                     out(q.nbytes());
  const kvb::KvView view = pool.layer_view(0);
  const int n_cached = static_cast<int>(k.shape(0)), kv_dim = n_kv_heads * head_dim;
  DeviceBuffer stacked;
  if (stacked_kv) {
    std::vector<f16> host(static_cast<std::size_t>(n_cached) * 2 * kv_dim);
    for (std::size_t t = 0; t < k.shape(0); ++t) {
      std::copy_n(k.data() + t * kv_dim, kv_dim, host.begin() + 2 * t * kv_dim);
      std::copy_n(v.data() + t * kv_dim, kv_dim, host.begin() + (2 * t + 1) * kv_dim);
    }
    stacked = DeviceBuffer(host.size() * sizeof(f16));
    KVB_CUDA(cudaMemcpy(stacked.as(), host.data(), stacked.bytes(), cudaMemcpyHostToDevice));
    kvb::kernels::kv_write(stacked.as<f16>(), stacked.as<f16>() + kv_dim, dr.as<StepRow>(), view,
                           n_cached, n_kv_heads, head_dim, 2 * kv_dim, nullptr);
  } else {
    kvb::kernels::kv_write(dk.as<f16>(), dv.as<f16>(), dr.as<StepRow>(), view, n_cached, n_kv_heads,
                           head_dim, kv_dim, nullptr);
  }
  const int n_query = n_tokens - query_from;
  const f16* query = dq.as<f16>() + static_cast<std::size_t>(query_from) * n_heads * head_dim;
  const StepRow* query_rows = dr.as<StepRow>() + query_from;
  if (kernel == "decode") {
    kvb::kernels::attention(query, out.as<f16>(), query_rows, view, n_query, n_heads, n_kv_heads,
                            head_dim, max_ctx, fixed_ctx,
                            wide_key_loads ? kvb::kernels::widest_key_loads(view, head_dim)
                                           : kvb::kernels::KeyLoads::Scalar,
                            nullptr);
  } else {
    const DeviceTiles tiles(std::span(rows).subspan(query_from));
    if (kernel == "prefill") {
      kvb::kernels::attention_prefill(query, out.as<f16>(), query_rows,
                                      tiles.data.as<kvb::AttnTile>(), tiles.count, view, n_heads,
                                      n_kv_heads, head_dim, nullptr);
    } else {
      kvb::kernels::attention_dense(
          query, out.as<f16>(), query_rows, tiles.data.as<kvb::AttnTile>(), tiles.count, view,
          n_heads * head_dim, fixed_ctx, n_heads, n_kv_heads, head_dim, nullptr);
    }
  }
  return download<f16>(out.as(), {static_cast<std::size_t>(n_query), q.shape(1), q.shape(2)});
}

// An encoder's self attention on its stacked product: qkv [n_seqs * n_keys, 3 * dim] holds each
// row's q, k and v side by side, and every row attends all n_keys rows of its sequence. Keys
// and values are read where they are, through a cache view laid over qkv. Returns [rows, dim].
Out<f16> attention_dense_stacked(In<f16> qkv, int n_seqs, int n_heads) {
  const int n_rows = static_cast<int>(qkv.shape(0)), dim = static_cast<int>(qkv.shape(1)) / 3;
  const int n_keys = n_rows / n_seqs, head_dim = dim / n_heads;
  std::vector<StepRow> rows;
  for (int r = 0; r < n_rows; ++r) rows.push_back({.position = r % n_keys, .seq = r / n_keys});
  std::vector<std::int32_t> blocks(n_seqs);
  std::iota(blocks.begin(), blocks.end(), 0);
  const DeviceTiles tiles(rows);
  const DeviceBuffer dq = upload(qkv), dr = upload_rows(rows),
                     db(blocks.size() * sizeof(std::int32_t)),
                     out(static_cast<std::size_t>(n_rows) * dim * sizeof(f16));
  KVB_CUDA(cudaMemcpy(db.as(), blocks.data(), db.bytes(), cudaMemcpyHostToDevice));
  const kvb::KvView view{
      .k = dq.as<f16>() + dim,
      .v = dq.as<f16>() + 2 * dim,
      .block_table = db.as<std::int32_t>(),
      .blocks_per_seq = 1,
      .block_size = n_keys,
      .block_stride = static_cast<std::int64_t>(n_keys) * 3 * dim,
      .head_stride = head_dim,
      .pos_stride = 3 * dim,
      .dim_stride = 1,
  };
  kvb::kernels::attention_dense(dq.as<f16>(), out.as<f16>(), dr.as<StepRow>(),
                                tiles.data.as<kvb::AttnTile>(), tiles.count, view, 3 * dim, n_keys,
                                n_heads, n_heads, head_dim, nullptr);
  return download<f16>(out.as(), {qkv.shape(0), static_cast<std::size_t>(dim)});
}

// --- Engine
// ---------------------------------------------------------------------------------------

// The engine with a teacher-forcing entry point: feed tokens of one sequence and read back every
// capture site. feed() gives row r the blocks r * blocks_per_seq... in order.
class PyEngine {
 public:
  PyEngine(const std::string& model_dir, int max_ctx, int max_seqs, int max_step_tokens,
           int kv_block_size, const std::string& kv_layout, const std::string& gemm, bool graph,
           const std::string& plan, bool wide_key_loads, int kv_blocks, int max_graphs,
           const std::string& kv_dtype, int quantized_gemm_rows, std::size_t dequant_slice_bytes,
           bool fuse, int f16_gemv_rows, bool expand_quantized, bool prefill_attention)
      : engine_(model_dir, kvb::cuda::EngineConfig{
                               .max_ctx = max_ctx,
                               .max_seqs = max_seqs,
                               .max_step_tokens = max_step_tokens,
                               // feed() asks for the logits of every row.
                               .max_outputs = max_step_tokens > 0 ? max_step_tokens : max_ctx,
                               .kv_block_size = kv_block_size,
                               .kv_blocks = kv_blocks,
                               .kv_layout = layout_from(kv_layout),
                               .kv_dtype = kv_dtype_from(kv_dtype),
                               .plan = plan == "naive" ? kvb::PlanStrategy::Naive
                                                       : kvb::PlanStrategy::GreedyBySize,
                               .gemm = accumulate_from(gemm),
                               .fuse = fuse,
                               .f16_gemv_rows = f16_gemv_rows,
                               .wide_key_loads = wide_key_loads,
                               .prefill_attention = prefill_attention,
                               .quantized_gemm_rows = quantized_gemm_rows,
                               .expand_quantized = expand_quantized,
                               .dequant_slice_bytes = dequant_slice_bytes,
                               .use_graph = graph,
                               .max_graphs = max_graphs,
                           }) {}

  // Processes `ids` as positions start, start + 1, ... of block-table row `seq`, with every row
  // an output. Returns logits [n, vocab] and, when capture is set, every capture site.
  std::map<std::string, Out<float>> feed(const std::vector<std::int32_t>& ids, int start, int seq,
                                         bool capture) {
    const int block_size = engine_.kv_block_size();
    const int blocks_per_seq = (engine_.config().max_ctx + block_size - 1) / block_size;
    // generate() leaves the row pointing at whatever blocks its sequence was given.
    std::vector<int> blocks;
    for (int i = 0; i < blocks_per_seq; ++i) blocks.push_back(seq * blocks_per_seq + i);
    engine_.set_blocks(seq, blocks);

    std::vector<StepRow> rows;
    std::vector<std::int32_t> out_rows;
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
      rows.push_back({.token_id = ids[i],
                      .position = start + i,
                      .seq = seq,
                      .kv_slot = seq * blocks_per_seq * block_size + start + i});
      out_rows.push_back(i);
    }
    kvb::cuda::Capture sites;
    engine_.forward(rows, out_rows, {.capture = capture ? &sites : nullptr});
    if (!capture) sites["logits"] = engine_.logits();

    std::map<std::string, Out<float>> result;
    for (const auto& [name, values] : sites) {
      result.emplace(name, to_numpy(values, {ids.size(), values.size() / ids.size()}));
    }
    return result;
  }

  std::vector<std::int32_t> generate(const std::vector<std::int32_t>& prompt, int max_new_tokens,
                                     float temperature, int top_k, float top_p, std::uint64_t seed,
                                     std::int32_t eos_token_id) {
    return kvb::cuda::generate(
        engine_, prompt,
        {.max_new_tokens = max_new_tokens,
         .sampler = {.temperature = temperature, .top_k = top_k, .top_p = top_p, .seed = seed},
         .eos_token_id = eos_token_id});
  }

  // Serves the prompts over the engine's max_seqs block-table rows under the given scheduling
  // policy; returns each request's new tokens, in request order. arrivals_s, when given, holds
  // one arrival time per prompt.
  std::vector<std::vector<std::int32_t>> serve(
      const std::vector<std::vector<std::int32_t>>& prompts, const std::vector<int>& max_new_tokens,
      const std::string& policy, const std::vector<double>& arrivals_s) {
    if (prompts.size() != max_new_tokens.size()) throw kvb::Error("one length per prompt");
    if (!arrivals_s.empty() && arrivals_s.size() != prompts.size()) {
      throw kvb::Error("one arrival time per prompt");
    }
    const auto parsed = kvb::schedule_policy_from_name(policy);
    if (!parsed) throw kvb::Error("unknown scheduling policy " + policy);
    std::vector<kvb::cuda::ServeRequest> requests;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      requests.push_back({.id = static_cast<int>(i),
                          .prompt = prompts[i],
                          .max_new_tokens = max_new_tokens[i],
                          .arrival_s = arrivals_s.empty() ? 0.0 : arrivals_s[i]});
    }
    std::vector<std::vector<std::int32_t>> tokens(prompts.size());
    serve_stats_ = {};
    kvb::cuda::serve(engine_, requests, {.policy = *parsed}, [&](const kvb::cuda::ServeStep& step) {
      serve_stats_["steps"] += 1;
      serve_stats_["preemptions"] += static_cast<int>(step.plan.preempt.size());
      serve_stats_["max_running"] = std::max(serve_stats_["max_running"], step.usage.running);
      for (const kvb::StepResult::Token& token : step.result.tokens) {
        tokens[token.seq].push_back(token.token);
      }
    });
    return tokens;
  }

  // "fp16", "int8_pc" or "int4_g": how the loaded checkpoint stores its matrices.
  std::string weights() const {
    return std::string(kvb::weight_scheme_name(engine_.weight_format().scheme));
  }
  // Device bytes: weights, kv_pool, arena, dequant.
  std::map<std::string, std::size_t> memory() const {
    const kvb::cuda::EngineMemory m = engine_.memory();
    return {
        {"weights", m.weights}, {"kv_pool", m.kv_pool}, {"arena", m.arena}, {"dequant", m.dequant}};
  }

  // Counters of the latest serve(): steps, preemptions, max_running.
  std::map<std::string, int> serve_stats() const { return serve_stats_; }
  // Graphs the engine holds launch counters for: replayed, eager, captured, evicted.
  std::map<std::string, std::int64_t> graph_stats() const {
    const kvb::cuda::GraphStats& stats = engine_.graph_stats();
    return {{"replayed", stats.replayed},
            {"eager", stats.eager},
            {"captured", stats.captured},
            {"evicted", stats.evicted}};
  }

 private:
  kvb::cuda::Engine engine_;
  std::map<std::string, int> serve_stats_;
};

// --- AsrEngine
// ------------------------------------------------------------------------------------

// Captured sites of a step of `program` as arrays [rows, columns of the site's value].
std::map<std::string, Out<float>> site_arrays(const kvb::cuda::Capture& sites,
                                              const kvb::Program& program) {
  std::map<std::string, std::size_t> cols;
  for (const kvb::Op& op : program.ops) {
    if (!op.in0_site.empty()) cols[op.in0_site] = program.values[op.in0].cols;
    if (!op.site.empty()) cols[op.site] = program.values[op.out].cols;
  }
  std::map<std::string, Out<float>> result;
  for (const auto& [name, values] : sites) {
    const std::size_t n_cols = cols.at(name);
    result.emplace(name, to_numpy(values, {values.size() / n_cols, n_cols}));
  }
  return result;
}

// The speech engine with its two raw steps: encode clips into the Cross cache of block-table
// rows, and feed tokens of one sequence (teacher forcing). feed() gives row r the Self blocks
// r * blocks_per_seq... in order.
class PyAsrEngine {
 public:
  PyAsrEngine(const std::string& model_dir, int max_seqs, int max_ctx, int max_step_tokens,
              int kv_block_size, int kv_blocks, const std::string& kv_layout,
              const std::string& kv_dtype, const std::string& cross_kv_dtype, int max_encode_batch,
              const std::string& gemm, bool graph, int max_graphs, const std::string& plan,
              bool wide_key_loads, int quantized_gemm_rows, std::size_t dequant_slice_bytes,
              bool fuse, int f16_gemv_rows, bool expand_quantized, bool prefill_attention)
      : engine_(model_dir,
                kvb::cuda::EngineConfig{
                    .max_ctx = max_ctx,
                    .max_seqs = max_seqs,
                    .max_step_tokens = max_step_tokens,
                    // feed() asks for the logits of every row: as many as a step has.
                    .max_outputs = std::numeric_limits<int>::max(),
                    .kv_block_size = kv_block_size,
                    .kv_blocks = kv_blocks,
                    .kv_layout = layout_from(kv_layout),
                    .kv_dtype = kv_dtype_from(kv_dtype),
                    .plan = plan == "naive" ? kvb::PlanStrategy::Naive
                                            : kvb::PlanStrategy::GreedyBySize,
                    .gemm = accumulate_from(gemm),
                    .fuse = fuse,
                    .f16_gemv_rows = f16_gemv_rows,
                    .wide_key_loads = wide_key_loads,
                    .prefill_attention = prefill_attention,
                    .quantized_gemm_rows = quantized_gemm_rows,
                    .expand_quantized = expand_quantized,
                    .dequant_slice_bytes = dequant_slice_bytes,
                    .use_graph = graph,
                    .max_graphs = max_graphs,
                },
                kvb::cuda::AsrOptions{.max_encode_batch = max_encode_batch,
                                      .cross_kv_dtype = kv_dtype_from(cross_kv_dtype)}),
        plan_(plan) {}

  // Encodes clips in one encoder step: features [n_mels, n_frames] or [clips, n_mels,
  // n_frames], clip i into the Cross block of block-table row seqs[i] (by default row i).
  // Returns every "encoder.*" site when capture is set, the clips' rows end to end.
  std::map<std::string, Out<float>> encode(In<float> features,
                                           const std::optional<std::vector<int>>& seqs,
                                           bool capture) {
    if (features.ndim() != 2 && features.ndim() != 3) {
      throw kvb::Error("features must be [n_mels, n_frames] or [clips, n_mels, n_frames]");
    }
    const kvb::WhisperConfig& cfg = engine_.model_config();
    const std::size_t n_clips = features.ndim() == 3 ? features.shape(0) : 1;
    const std::size_t first = features.ndim() - 2;
    if (static_cast<std::int64_t>(features.shape(first)) != cfg.n_mels ||
        static_cast<std::int64_t>(features.shape(first + 1)) != cfg.n_frames()) {
      throw kvb::Error("each clip needs features [" + std::to_string(cfg.n_mels) + ", " +
                       std::to_string(cfg.n_frames()) + "]");
    }
    std::vector<int> rows(n_clips);
    std::iota(rows.begin(), rows.end(), 0);
    kvb::cuda::Capture sites;
    engine_.encode({features.data(), features.size()}, seqs ? *seqs : rows,
                   {.capture = capture ? &sites : nullptr});
    return site_arrays(sites, engine_.programs().encoder);
  }

  // Processes `ids` as positions start, start + 1, ... of block-table row `seq`, whose clip
  // must have been encoded, with every row an output. With capture set, returns every
  // "decoder.*" site and the logits [n, vocab] as computed; otherwise the logits alone, with
  // the tokens the rules suppress at -inf.
  std::map<std::string, Out<float>> feed(const std::vector<std::int32_t>& ids, int start, int seq,
                                         bool capture) {
    const int block_size = engine_.kv_block_size();
    const int blocks_per_seq = (engine_.config().max_ctx + block_size - 1) / block_size;
    if (seq < 0 || (seq + 1) * blocks_per_seq > engine_.config().kv_blocks) {
      throw kvb::Error("feed() has no Self blocks for row " + std::to_string(seq));
    }
    std::vector<int> blocks;
    for (int i = 0; i < blocks_per_seq; ++i) blocks.push_back(seq * blocks_per_seq + i);
    engine_.set_blocks(seq, blocks);

    std::vector<StepRow> rows;
    std::vector<std::int32_t> out_rows;
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
      rows.push_back({.token_id = ids[i],
                      .position = start + i,
                      .seq = seq,
                      .kv_slot = seq * blocks_per_seq * block_size + start + i});
      out_rows.push_back(i);
    }
    kvb::cuda::Capture sites;
    engine_.forward(rows, out_rows, {.capture = capture ? &sites : nullptr});
    if (!capture) sites["logits"] = engine_.logits();
    return site_arrays(sites, engine_.programs().decoder);
  }

  // Greedy token of each row of the last feed(), after the suppress rules.
  std::vector<std::int32_t> next_ids() const {
    const std::span<const std::int32_t> ids = engine_.next_ids();
    return {ids.begin(), ids.end()};
  }

  // "fp16", "int8_pc" or "int4_g": how the loaded checkpoint stores its matrices.
  std::string weights() const {
    return std::string(kvb::weight_scheme_name(engine_.weight_format().scheme));
  }

  // Every engine option as the engine resolved it, for result records.
  nb::dict config() const {
    const kvb::cuda::EngineConfig& c = engine_.config();
    const kvb::cuda::AsrOptions& o = engine_.options();
    nb::dict d;
    d["max_seqs"] = c.max_seqs;
    d["max_ctx"] = c.max_ctx;
    d["max_step_tokens"] = c.max_step_tokens;
    d["max_outputs"] = c.max_outputs;
    d["kv_block_size"] = c.kv_block_size;
    d["kv_blocks"] = c.kv_blocks;
    d["kv_layout"] = std::string(kvb::kv_layout_name(c.kv_layout));
    d["kv_dtype"] = std::string(kvb::kv_dtype_name(c.kv_dtype));
    d["cross_kv_dtype"] = std::string(kvb::kv_dtype_name(o.cross_kv_dtype));
    d["max_encode_batch"] = o.max_encode_batch;
    d["gemm"] = std::string(kvb::cuda::gemm_accumulate_name(c.gemm));
    d["graph"] = c.use_graph;
    d["max_graphs"] = c.max_graphs;
    d["plan"] = plan_;
    d["wide_key_loads"] = c.wide_key_loads;
    d["quantized_gemm_rows"] = c.quantized_gemm_rows;
    d["dequant_slice_bytes"] = c.dequant_slice_bytes;
    d["fuse"] = c.fuse;
    d["f16_gemv_rows"] = c.f16_gemv_rows;
    d["expand_quantized"] = c.expand_quantized;
    d["prefill_attention"] = c.prefill_attention;
    d["weights"] = weights();
    d["weight_group_size"] = engine_.weight_format().group_size;
    return d;
  }

  // The decoding rules the engine read from generation.json.
  nb::dict rules() const {
    const kvb::WhisperRules& r = engine_.rules();
    nb::dict d;
    d["prompt"] = r.prompt;
    d["suppress_tokens"] = r.suppress_tokens;
    d["begin_suppress_tokens"] = r.begin_suppress_tokens;
    d["eos_token_id"] = r.eos_token_id;
    d["max_length"] = r.max_length;
    return d;
  }

  // Device bytes: weights, self_kv, cross_kv, arena, encoder_arena, decoder_arena, dequant.
  std::map<std::string, std::size_t> memory() const {
    const kvb::cuda::AsrMemory m = engine_.memory();
    return {{"weights", m.weights},
            {"self_kv", m.self_kv},
            {"cross_kv", m.cross_kv},
            {"arena", m.arena},
            {"encoder_arena", m.encoder_arena},
            {"decoder_arena", m.decoder_arena},
            {"dequant", m.dequant}};
  }

 private:
  kvb::cuda::AsrEngine engine_;
  std::string plan_;
};

}  // namespace

NB_MODULE(_kvb, m) {
  m.def("rms_norm", &rms_norm, "x"_a, "weight"_a, "eps"_a);
  m.def("layer_norm", &layer_norm, "x"_a, "weight"_a, "bias"_a, "eps"_a);
  m.def("add_layer_norm", &add_layer_norm, "x"_a, "y"_a, "weight"_a, "bias"_a, "eps"_a);
  m.def("gelu", &gelu, "x"_a);
  m.def("add_positions", &add_positions, "x"_a, "table"_a, "positions"_a);
  m.def("unfold3", &unfold3, "x"_a, "n_seqs"_a = 1, "stride"_a = 1);
  m.def("add_bias", &add_bias, "y"_a, "bias"_a);
  m.def("add_bias", &add_bias_f32, "y"_a, "bias"_a);
  m.def("suppress_logits", &suppress_logits, "logits"_a, "positions"_a, "out_rows"_a, "always"_a,
        "begin"_a, "begin_pos"_a);
  m.def("embedding", &embedding, "table"_a, "ids"_a);
  m.def("rope", &rope, "x"_a, "positions"_a, "cos_sin"_a);
  m.def("silu_mul", &silu_mul, "gate"_a, "up"_a);
  m.def("silu_mul_stacked", &silu_mul_stacked, "gate_up"_a);
  m.def("add_rms_norm", &add_rms_norm, "x"_a, "y"_a, "weight"_a, "eps"_a);
  m.def("add", &add, "x"_a, "y"_a);
  m.def("gather_rows", &gather_rows, "x"_a, "out_rows"_a);
  m.def("argmax", &argmax, "logits"_a);
  m.def("linear", &linear, "weight"_a, "x"_a, "accumulate"_a = "fp32", "f32_out"_a = false);
  m.def("linear_f16", &linear_f16, "weight"_a, "x"_a, "f32_out"_a = false, "bias"_a = nb::none());
  m.def("kv_attention", &kv_attention, "q"_a, "k"_a, "v"_a, "positions"_a, "seqs"_a, "kv_slots"_a,
        "block_table"_a, "block_size"_a, "n_blocks"_a, "layout"_a = "head_major",
        "query_from"_a = 0, "wide_key_loads"_a = true, "kv_dtype"_a = "fp16", "kernel"_a = "decode",
        "fixed_ctx"_a = 0, "stacked_kv"_a = false);
  m.def("attention_dense_stacked", &attention_dense_stacked, "qkv"_a, "n_seqs"_a, "n_heads"_a);
  m.def("linear_quantized", &linear_quantized, "scheme"_a, "data"_a, "scale"_a, "x"_a,
        "group_size"_a = 0, "f32_out"_a = false, "bias"_a = nb::none());
  m.def("gemm_quantized", &gemm_quantized, "scheme"_a, "data"_a, "scale"_a, "x"_a,
        "group_size"_a = 0, "f32_out"_a = false, "bias"_a = nb::none());
  m.def("run_linear", &run_linear, "scheme"_a, "data"_a, "scale"_a, "x"_a, "bias"_a = nb::none(),
        "group_size"_a = 0, "f32_out"_a = false, "f16_gemv_rows"_a = 4, "quantized_gemm_rows"_a = 3,
        "expand_quantized"_a = false, "dequant_slice_bytes"_a = std::size_t{32} << 20,
        "accumulate"_a = "fp32");
  m.def("dequantize", &dequantize, "scheme"_a, "data"_a, "scale"_a, "group_size"_a = 0,
        "first_channel"_a = 0, "n_channels"_a = 0);
  m.def("embedding_quantized", &embedding_quantized, "scheme"_a, "data"_a, "scale"_a, "ids"_a,
        "group_size"_a = 0);

  nb::class_<PyEngine>(m, "Engine")
      .def(nb::init<const std::string&, int, int, int, int, const std::string&, const std::string&,
                    bool, const std::string&, bool, int, int, const std::string&, int, std::size_t,
                    bool, int, bool, bool>(),
           "model_dir"_a, "max_ctx"_a = 2048, "max_seqs"_a = 1, "max_step_tokens"_a = 0,
           "kv_block_size"_a = 0, "kv_layout"_a = "head_major", "gemm"_a = "fp32", "graph"_a = true,
           "plan"_a = "greedy", "wide_key_loads"_a = true, "kv_blocks"_a = 0, "max_graphs"_a = 16,
           "kv_dtype"_a = "fp16", "quantized_gemm_rows"_a = 3,
           "dequant_slice_bytes"_a = std::size_t{32} << 20, "fuse"_a = true, "f16_gemv_rows"_a = 4,
           "expand_quantized"_a = false, "prefill_attention"_a = true)
      .def("feed", &PyEngine::feed, "ids"_a, "start"_a = 0, "seq"_a = 0, "capture"_a = false)
      .def("generate", &PyEngine::generate, "prompt"_a, "max_new_tokens"_a, "temperature"_a = 0.0f,
           "top_k"_a = 0, "top_p"_a = 1.0f, "seed"_a = 0, "eos_token_id"_a = -1)
      .def("serve", &PyEngine::serve, "prompts"_a, "max_new_tokens"_a, "policy"_a = "static",
           "arrivals_s"_a = std::vector<double>{})
      .def("weights", &PyEngine::weights)
      .def("memory", &PyEngine::memory)
      .def("serve_stats", &PyEngine::serve_stats)
      .def("graph_stats", &PyEngine::graph_stats);

  nb::class_<PyAsrEngine>(m, "AsrEngine")
      .def(nb::init<const std::string&, int, int, int, int, int, const std::string&,
                    const std::string&, const std::string&, int, const std::string&, bool, int,
                    const std::string&, bool, int, std::size_t, bool, int, bool, bool>(),
           "model_dir"_a, "max_seqs"_a = 1, "max_ctx"_a = 0, "max_step_tokens"_a = 0,
           "kv_block_size"_a = 16, "kv_blocks"_a = 0, "kv_layout"_a = "head_major",
           "kv_dtype"_a = "fp16", "cross_kv_dtype"_a = "fp16", "max_encode_batch"_a = 1,
           "gemm"_a = "fp32", "graph"_a = true, "max_graphs"_a = 16, "plan"_a = "greedy",
           "wide_key_loads"_a = true, "quantized_gemm_rows"_a = 3,
           "dequant_slice_bytes"_a = std::size_t{32} << 20, "fuse"_a = true, "f16_gemv_rows"_a = 4,
           "expand_quantized"_a = false, "prefill_attention"_a = true)
      .def("encode", &PyAsrEngine::encode, "features"_a, "seqs"_a = nb::none(), "capture"_a = false)
      .def("feed", &PyAsrEngine::feed, "ids"_a, "start"_a = 0, "seq"_a = 0, "capture"_a = false)
      .def("next_ids", &PyAsrEngine::next_ids)
      .def("weights", &PyAsrEngine::weights)
      .def("config", &PyAsrEngine::config)
      .def("rules", &PyAsrEngine::rules)
      .def("memory", &PyAsrEngine::memory);
}
