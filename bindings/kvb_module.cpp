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

// Writes k and v [tokens, kv_heads, head_dim] into a fresh cache at each token's kv_slot, then
// attends q [tokens, heads, head_dim] over it for the tokens from query_from on.
// block_table is [sequences, blocks_per_seq].
Out<f16> kv_attention(In<f16> q, In<f16> k, In<f16> v, In<std::int32_t> positions,
                      In<std::int32_t> seqs, In<std::int32_t> kv_slots,
                      In<std::int32_t> block_table, int block_size, int n_blocks,
                      const std::string& layout, int query_from) {
  const int n_tokens = static_cast<int>(q.shape(0)), n_heads = static_cast<int>(q.shape(1));
  const int head_dim = static_cast<int>(q.shape(2)), n_kv_heads = static_cast<int>(k.shape(1));
  const int blocks_per_seq = static_cast<int>(block_table.shape(1));
  kvb::cuda::KvPool pool(kvb::KvDims{.n_layers = 1,
                                     .n_blocks = n_blocks,
                                     .n_kv_heads = n_kv_heads,
                                     .block_size = block_size,
                                     .head_dim = head_dim},
                         layout_from(layout), static_cast<int>(block_table.shape(0)),
                         blocks_per_seq);
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
                          n_kv_heads, head_dim, max_ctx, nullptr);
  return download<f16>(out.as(), {static_cast<std::size_t>(n_query), q.shape(1), q.shape(2)});
}

// --- Engine
// ---------------------------------------------------------------------------------------

// The engine with a teacher-forcing entry point: feed tokens of one sequence and read back every
// capture site. Blocks are assigned to rows in order, so row r owns blocks r * blocks_per_seq...
class PyEngine {
 public:
  PyEngine(const std::string& model_dir, int max_ctx, int max_seqs, int max_step_tokens,
           int kv_block_size, const std::string& kv_layout, const std::string& gemm, bool graph,
           const std::string& plan)
      : engine_(model_dir, kvb::cuda::EngineConfig{
                               .max_ctx = max_ctx,
                               .max_seqs = max_seqs,
                               .max_step_tokens = max_step_tokens,
                               .kv_block_size = kv_block_size,
                               .kv_layout = layout_from(kv_layout),
                               .plan = plan == "naive" ? kvb::PlanStrategy::Naive
                                                       : kvb::PlanStrategy::GreedyBySize,
                               .gemm = accumulate_from(gemm),
                               .use_graph = graph,
                           }) {
    const int block_size = engine_.kv_block_size();
    const int blocks_per_seq = (engine_.config().max_ctx + block_size - 1) / block_size;
    for (int seq = 0; seq < engine_.config().max_seqs; ++seq) {
      std::vector<int> blocks;
      for (int i = 0; i < blocks_per_seq; ++i) blocks.push_back(seq * blocks_per_seq + i);
      engine_.set_blocks(seq, blocks);
    }
  }

  // Processes `ids` as positions start, start + 1, ... of block-table row `seq`, with every row
  // an output. Returns logits [n, vocab] and, when capture is set, every capture site.
  std::map<std::string, Out<float>> feed(const std::vector<std::int32_t>& ids, int start, int seq,
                                         bool capture) {
    const int block_size = engine_.kv_block_size();
    const int blocks_per_seq = (engine_.config().max_ctx + block_size - 1) / block_size;
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

  std::vector<std::int32_t> last_argmax() const {
    const auto ids = engine_.next_ids();
    return {ids.begin(), ids.end()};
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

  std::map<std::string, std::size_t> memory() const {
    const kvb::cuda::EngineMemory m = engine_.memory();
    return {{"weights", m.weights},
            {"kv_pool", m.kv_pool},
            {"arena", m.arena},
            {"arena_naive", m.arena_naive},
            {"arena_peak_live", m.arena_peak_live}};
  }

 private:
  kvb::cuda::Engine engine_;
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
        "query_from"_a = 0);

  nb::class_<PyEngine>(m, "Engine")
      .def(nb::init<const std::string&, int, int, int, int, const std::string&, const std::string&,
                    bool, const std::string&>(),
           "model_dir"_a, "max_ctx"_a = 2048, "max_seqs"_a = 1, "max_step_tokens"_a = 0,
           "kv_block_size"_a = 0, "kv_layout"_a = "head_major", "gemm"_a = "fp32", "graph"_a = true,
           "plan"_a = "greedy")
      .def("feed", &PyEngine::feed, "ids"_a, "start"_a = 0, "seq"_a = 0, "capture"_a = false)
      .def("last_argmax", &PyEngine::last_argmax)
      .def("generate", &PyEngine::generate, "prompt"_a, "max_new_tokens"_a, "temperature"_a = 0.0f,
           "top_k"_a = 0, "top_p"_a = 1.0f, "seed"_a = 0, "eos_token_id"_a = -1)
      .def("memory", &PyEngine::memory);
}
