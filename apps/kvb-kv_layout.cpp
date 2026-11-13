// KV-cache layout benchmark: the engine's own kv_write and attention kernels on pools laid out
// three ways, at decode shapes (one query per sequence, long context) and one prefill shape.
//
// A step is what one decode step does to the cache: for every layer, store the new keys and
// values, then attend over everything cached. Layers are visited in order, as in a forward pass,
// so each layer's keys have left the caches by the time they are read again.
//
// Prints one JSON object on stdout; `python -m kvbound.microbench kv_layout` records it.
// Options restrict the run to some points, so a profiler can look at single kernel launches:
//   --layout NAME --key-loads NAME --shape decode|prefill --batch N --ctx N --reps N

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "kvbound/cuda/device.h"
#include "kvbound/cuda/kernels.h"
#include "kvbound/cuda/kv_pool.h"
#include "kvbound/error.h"

namespace {

using kvb::StepRow;
using kvb::cuda::DeviceBuffer;
using kvb::kernels::f16;
using kvb::kernels::KeyLoads;

// TinyLlama 1.1B's attention shape.
constexpr int kLayers = 22;
constexpr int kHeads = 32;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 64;
constexpr int kMaxCtx = 4096;
constexpr int kMaxBatch = 8;
constexpr int kPrefillTokens = 512;
constexpr int kReps = 30;

// Every byte 0x3C makes every FP16 element 0x3C3C, about 1.06: finite in all the arithmetic.
constexpr int kFillByte = 0x3C;

// Which points run. An empty name or a zero matches every value.
struct Options {
  std::string layout, key_loads, shape;
  int batch = 0;
  int ctx = 0;
  int reps = kReps;

  Options(int argc, char** argv) {
    if (argc % 2 == 0) throw kvb::Error("every option needs a value");
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string key = argv[i], value = argv[i + 1];
      if (key == "--layout") {
        if (!kvb::kv_layout_from_name(value)) throw kvb::Error("unknown layout " + value);
        layout = value;
      } else if (key == "--key-loads") {
        if (value != "scalar" && value != "wide") throw kvb::Error("unknown key loads " + value);
        key_loads = value;
      } else if (key == "--shape") {
        if (value != "decode" && value != "prefill") throw kvb::Error("unknown shape " + value);
        shape = value;
      } else if (key == "--batch") {
        batch = integer(key, value);
      } else if (key == "--ctx") {
        ctx = integer(key, value);
      } else if (key == "--reps") {
        reps = integer(key, value);
      } else {
        throw kvb::Error("unknown option " + key);
      }
    }
    if (reps < 1) throw kvb::Error("--reps must be at least 1");
  }

  static int integer(const std::string& key, const std::string& value) {
    std::size_t used = 0;
    int parsed = 0;
    try {
      parsed = std::stoi(value, &used);
    } catch (const std::exception&) {
      used = 0;
    }
    if (used == 0 || used != value.size()) {
      throw kvb::Error(key + " needs an integer, got " + value);
    }
    return parsed;
  }
  static bool matches(const std::string& wanted, std::string_view value) {
    return wanted.empty() || wanted == value;
  }
  bool wants_layout(kvb::KvLayout l) const { return matches(layout, kvb::kv_layout_name(l)); }
  bool wants(KeyLoads loads, std::string_view point_shape, int point_batch, int point_ctx) const {
    return matches(key_loads, kvb::kernels::key_loads_name(loads)) && matches(shape, point_shape) &&
           (batch == 0 || batch == point_batch) && (ctx == 0 || ctx == point_ctx);
  }
};

template <typename F>
double seconds(F&& work) {
  KVB_CUDA(cudaDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  work();
  KVB_CUDA(cudaDeviceSynchronize());
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct Timing {
  double best_ms;
  double median_ms;
};

template <typename F>
Timing time_steps(int reps, F&& step) {
  step();  // warm-up
  std::vector<double> ms;
  for (int i = 0; i < reps; ++i) ms.push_back(1e3 * seconds(step));
  std::sort(ms.begin(), ms.end());
  return {ms.front(), ms[ms.size() / 2]};
}

DeviceBuffer filled(std::size_t bytes) {
  DeviceBuffer buffer(bytes);
  KVB_CUDA(cudaMemset(buffer.as(), kFillByte, bytes));
  return buffer;
}

DeviceBuffer upload(const std::vector<StepRow>& rows) {
  DeviceBuffer buffer(rows.size() * sizeof(StepRow));
  KVB_CUDA(cudaMemcpy(buffer.as(), rows.data(), buffer.bytes(), cudaMemcpyHostToDevice));
  return buffer;
}

int run(const Options& options) {
  const std::size_t q_bytes = static_cast<std::size_t>(kPrefillTokens) * kHeads * kHeadDim * 2;
  const DeviceBuffer q = filled(q_bytes), out = filled(q_bytes);
  const DeviceBuffer k = filled(q_bytes), v = filled(q_bytes);  // more than kv_write reads

  std::printf("{\n  \"benchmark\": \"kv_layout\",\n");
  std::printf(
      "  \"params\": {\"n_layers\": %d, \"n_heads\": %d, \"n_kv_heads\": %d, \"head_dim\": %d, "
      "\"max_ctx\": %d, \"slots\": %d, \"prefill_tokens\": %d, \"reps\": %d},\n  \"points\": [",
      kLayers, kHeads, kKvHeads, kHeadDim, kMaxCtx, kMaxBatch, kPrefillTokens, options.reps);

  bool first = true;
  for (const kvb::KvLayout layout :
       {kvb::KvLayout::HeadMajor, kvb::KvLayout::TokenMajor, kvb::KvLayout::DimMajor}) {
    if (!options.wants_layout(layout)) continue;
    kvb::cuda::KvPool pool(kvb::KvDims{.n_layers = kLayers,
                                       .n_blocks = kMaxBatch,
                                       .n_kv_heads = kKvHeads,
                                       .block_size = kMaxCtx,
                                       .head_dim = kHeadDim},
                           layout, kMaxBatch, 1);
    for (int seq = 0; seq < kMaxBatch; ++seq) {
      const int block = seq;
      pool.set_blocks(seq, {&block, 1}, nullptr);
    }
    // The pools start as the same finite value everywhere.
    for (int layer = 0; layer < kLayers; ++layer) {
      const kvb::KvView view = pool.layer_view(layer);
      const std::size_t layer_bytes = pool.bytes() / 2 / kLayers;
      if (layout != kvb::KvLayout::TokenMajor) {
        KVB_CUDA(cudaMemset(view.k, kFillByte, layer_bytes));
        KVB_CUDA(cudaMemset(view.v, kFillByte, layer_bytes));
      } else if (layer == 0) {
        // Token-major interleaves the layers, so the whole pool is one region.
        KVB_CUDA(cudaMemset(view.k, kFillByte, pool.bytes() / 2));
        KVB_CUDA(cudaMemset(view.v, kFillByte, pool.bytes() / 2));
      }
    }

    std::vector<KeyLoads> variants = {KeyLoads::Scalar};
    if (kvb::kernels::widest_key_loads(pool.layer_view(0), kHeadDim) == KeyLoads::Wide) {
      variants.push_back(KeyLoads::Wide);
    }
    for (const KeyLoads loads : variants) {
      const auto point = [&](const char* shape, int batch, int ctx, int n_tokens, int reps,
                             const std::vector<StepRow>& rows) {
        if (!options.wants(loads, shape, batch, ctx)) return;
        const DeviceBuffer device_rows = upload(rows);
        const StepRow* r = device_rows.as<StepRow>();
        const Timing write = time_steps(reps, [&] {
          for (int layer = 0; layer < kLayers; ++layer) {
            kvb::kernels::kv_write(k.as<f16>(), v.as<f16>(), r, pool.layer_view(layer), n_tokens,
                                   kKvHeads, kHeadDim, nullptr);
          }
        });
        const Timing attend = time_steps(reps, [&] {
          for (int layer = 0; layer < kLayers; ++layer) {
            kvb::kernels::attention(q.as<f16>(), out.as<f16>(), r, pool.layer_view(layer), n_tokens,
                                    kHeads, kKvHeads, kHeadDim, kMaxCtx, loads, nullptr);
          }
        });
        std::printf(
            "%s\n    {\"layout\": \"%s\", \"key_loads\": \"%s\", \"shape\": \"%s\", "
            "\"batch\": %d, \"ctx\": %d, \"attention_ms\": %.4f, \"attention_median_ms\": %.4f, "
            "\"write_ms\": %.4f, \"write_median_ms\": %.4f}",
            first ? "" : ",", std::string(kvb::kv_layout_name(layout)).c_str(),
            std::string(kvb::kernels::key_loads_name(loads)).c_str(), shape, batch, ctx,
            attend.best_ms, attend.median_ms, write.best_ms, write.median_ms);
        first = false;
      };

      for (const int batch : {1, kMaxBatch}) {
        for (const int ctx : {128, 256, 512, 1024, 2048, 4096}) {
          std::vector<StepRow> rows;
          for (int seq = 0; seq < batch; ++seq) {
            rows.push_back({.token_id = 0,
                            .position = ctx - 1,
                            .seq = seq,
                            .kv_slot = seq * kMaxCtx + ctx - 1});
          }
          point("decode", batch, ctx, batch, options.reps, rows);
        }
      }
      // Prefill: every position of one prompt attends to the positions before it.
      std::vector<StepRow> rows;
      for (int pos = 0; pos < kPrefillTokens; ++pos) {
        rows.push_back({.token_id = 0, .position = pos, .seq = 0, .kv_slot = pos});
      }
      point("prefill", 1, kPrefillTokens, kPrefillTokens, std::min(5, options.reps), rows);
    }
  }
  if (first) throw kvb::Error("no point matches the options");
  std::printf("\n  ]\n}\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(Options(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "kvb-kv_layout: %s\n", e.what());
    return 1;
  }
}
