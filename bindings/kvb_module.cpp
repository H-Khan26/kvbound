// Python bindings: each kernel as a function over NumPy arrays, and the Engine. Arrays cross
// the boundary on the host; FP16 data travels as uint16 (view it with .view(np.float16)).
// Meant for tests and evals, where a copy per call does not matter.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "kvbound/cuda/blas.h"
#include "kvbound/cuda/engine.h"
#include "kvbound/cuda/generate.h"
#include "kvbound/cuda/kernels.h"
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
  kvb::kernels::silu_mul(dg.as<f16>(), du.as<f16>(), gate.size(), nullptr);
  return download<f16>(dg.as(), {gate.size()});
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

// x [rows, n_in] times a quantized matrix stored as `data` [n_out, bytes per row] (int8 values
// or packed 4-bit pairs, as uint8) with FP16 `scale`. Returns float16 bits, or float32.
nb::object linear_quantized(const std::string& scheme, In<std::uint8_t> data, In<f16> scale,
                            In<f16> x, int group_size, bool f32_out) {
  const DeviceWeight w(scheme, data, scale, group_size);
  const int n_out = static_cast<int>(data.shape(0)), n_in = w.n_in(data);
  const int rows = static_cast<int>(x.shape(0));
  const std::size_t n = static_cast<std::size_t>(rows) * n_out;
  const DeviceBuffer dx = upload(x), out(n * (f32_out ? sizeof(float) : sizeof(f16)));
  kvb::kernels::linear_quantized(w.ref, dx.as<f16>(), out.as(), rows, n_in, n_out, f32_out,
                                 nullptr);
  if (f32_out) return nb::cast(download<float>(out.as(), {x.shape(0), data.shape(0)}));
  return nb::cast(download<f16>(out.as(), {x.shape(0), data.shape(0)}));
}

// The same product as linear_quantized on tensor cores, for wide steps.
nb::object gemm_quantized(const std::string& scheme, In<std::uint8_t> data, In<f16> scale,
                          In<f16> x, int group_size, bool f32_out) {
  const DeviceWeight w(scheme, data, scale, group_size);
  const int n_out = static_cast<int>(data.shape(0)), n_in = w.n_in(data);
  const int rows = static_cast<int>(x.shape(0));
  const std::size_t n = static_cast<std::size_t>(rows) * n_out;
  const DeviceBuffer dx = upload(x), out(n * (f32_out ? sizeof(float) : sizeof(f16)));
  kvb::kernels::gemm_quantized(w.ref, dx.as<f16>(), out.as(), rows, n_in, n_out, f32_out, nullptr);
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

// Writes k and v [tokens, kv_heads, head_dim] into a fresh cache at each token's kv_slot, then
// attends q [tokens, heads, head_dim] over it for the tokens from query_from on.
// block_table is [sequences, blocks_per_seq].
Out<f16> kv_attention(In<f16> q, In<f16> k, In<f16> v, In<std::int32_t> positions,
                      In<std::int32_t> seqs, In<std::int32_t> kv_slots,
                      In<std::int32_t> block_table, int block_size, int n_blocks,
                      const std::string& layout, int query_from, bool wide_key_loads,
                      const std::string& kv_dtype) {
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
  int max_ctx = 1;
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
  kvb::kernels::kv_write(dk.as<f16>(), dv.as<f16>(), dr.as<StepRow>(), view, n_tokens, n_kv_heads,
                         head_dim, nullptr);
  const int n_query = n_tokens - query_from;
  kvb::kernels::attention(dq.as<f16>() + static_cast<std::size_t>(query_from) * n_heads * head_dim,
                          out.as<f16>(), dr.as<StepRow>() + query_from, view, n_query, n_heads,
                          n_kv_heads, head_dim, max_ctx,
                          wide_key_loads ? kvb::kernels::widest_key_loads(view, head_dim)
                                         : kvb::kernels::KeyLoads::Scalar,
                          nullptr);
  return download<f16>(out.as(), {static_cast<std::size_t>(n_query), q.shape(1), q.shape(2)});
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
           const std::string& kv_dtype, int quantized_gemm_rows, std::size_t dequant_slice_bytes)
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
                               .wide_key_loads = wide_key_loads,
                               .quantized_gemm_rows = quantized_gemm_rows,
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

}  // namespace

NB_MODULE(_kvb, m) {
  m.def("rms_norm", &rms_norm, "x"_a, "weight"_a, "eps"_a);
  m.def("embedding", &embedding, "table"_a, "ids"_a);
  m.def("rope", &rope, "x"_a, "positions"_a, "cos_sin"_a);
  m.def("silu_mul", &silu_mul, "gate"_a, "up"_a);
  m.def("add", &add, "x"_a, "y"_a);
  m.def("gather_rows", &gather_rows, "x"_a, "out_rows"_a);
  m.def("argmax", &argmax, "logits"_a);
  m.def("linear", &linear, "weight"_a, "x"_a, "accumulate"_a = "fp32", "f32_out"_a = false);
  m.def("kv_attention", &kv_attention, "q"_a, "k"_a, "v"_a, "positions"_a, "seqs"_a, "kv_slots"_a,
        "block_table"_a, "block_size"_a, "n_blocks"_a, "layout"_a = "head_major",
        "query_from"_a = 0, "wide_key_loads"_a = true, "kv_dtype"_a = "fp16");
  m.def("linear_quantized", &linear_quantized, "scheme"_a, "data"_a, "scale"_a, "x"_a,
        "group_size"_a = 0, "f32_out"_a = false);
  m.def("gemm_quantized", &gemm_quantized, "scheme"_a, "data"_a, "scale"_a, "x"_a,
        "group_size"_a = 0, "f32_out"_a = false);
  m.def("dequantize", &dequantize, "scheme"_a, "data"_a, "scale"_a, "group_size"_a = 0,
        "first_channel"_a = 0, "n_channels"_a = 0);
  m.def("embedding_quantized", &embedding_quantized, "scheme"_a, "data"_a, "scale"_a, "ids"_a,
        "group_size"_a = 0);

  nb::class_<PyEngine>(m, "Engine")
      .def(nb::init<const std::string&, int, int, int, int, const std::string&, const std::string&,
                    bool, const std::string&, bool, int, int, const std::string&, int,
                    std::size_t>(),
           "model_dir"_a, "max_ctx"_a = 2048, "max_seqs"_a = 1, "max_step_tokens"_a = 0,
           "kv_block_size"_a = 0, "kv_layout"_a = "head_major", "gemm"_a = "fp32", "graph"_a = true,
           "plan"_a = "greedy", "wide_key_loads"_a = true, "kv_blocks"_a = 0, "max_graphs"_a = 16,
           "kv_dtype"_a = "fp16", "quantized_gemm_rows"_a = 16,
           "dequant_slice_bytes"_a = std::size_t{32} << 20)
      .def("feed", &PyEngine::feed, "ids"_a, "start"_a = 0, "seq"_a = 0, "capture"_a = false)
      .def("generate", &PyEngine::generate, "prompt"_a, "max_new_tokens"_a, "temperature"_a = 0.0f,
           "top_k"_a = 0, "top_p"_a = 1.0f, "seed"_a = 0, "eos_token_id"_a = -1)
      .def("serve", &PyEngine::serve, "prompts"_a, "max_new_tokens"_a, "policy"_a = "static",
           "arrivals_s"_a = std::vector<double>{})
      .def("weights", &PyEngine::weights)
      .def("memory", &PyEngine::memory)
      .def("serve_stats", &PyEngine::serve_stats)
      .def("graph_stats", &PyEngine::graph_stats);
}
