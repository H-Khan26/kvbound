// Runs clips through the speech engine and prints what it decoded as one JSON object on stdout.
// It is the binary kvbound.sanitize runs under compute-sanitizer; it times nothing.
//
// --clips names a safetensors file that holds, per clip, "<id>/input_features" (FP32 log-mel
// features [n_mels, n_frames]) and "<id>/input_ids" (I32 decoder ids): a golden file of a
// Whisper model.
//
// --mode feed  The clips are encoded --encode-batch at a time. Each clip's ids are then teacher
//              forced: all in one decoder step, then again one per step. Prints each row's
//              greedy token after the suppress rules.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "args.h"
#include "kvbound/cuda/asr_engine.h"
#include "kvbound/safetensors.h"

namespace {

using kvb::cuda::AsrEngine;

struct Clip {
  std::string id;
  std::vector<float> features;
  std::vector<std::int32_t> ids;
};

// Elements of a tensor as T, copied: a mapped tensor need not be aligned for T.
template <typename T>
std::vector<T> elements(const kvb::TensorView& tensor, kvb::DType dtype, const std::string& name) {
  if (tensor.dtype != dtype) throw kvb::Error(name + ": unexpected dtype");
  std::vector<T> values(tensor.nbytes / sizeof(T));
  std::memcpy(values.data(), tensor.data, values.size() * sizeof(T));
  return values;
}

std::vector<Clip> load_clips(const std::string& path) {
  const kvb::SafetensorsFile file{path};
  const std::string suffix = "/input_features";
  std::vector<Clip> clips;
  for (const auto& [name, tensor] : file.tensors()) {
    if (!name.ends_with(suffix)) continue;
    Clip clip{.id = name.substr(0, name.size() - suffix.size())};
    clip.features = elements<float>(tensor, kvb::DType::F32, name);
    const std::string ids = clip.id + "/input_ids";
    clip.ids = elements<std::int32_t>(file.at(ids), kvb::DType::I32, ids);
    clips.push_back(std::move(clip));
  }
  if (clips.empty()) throw kvb::Error(path + " holds no <id>/input_features tensor");
  return clips;
}

void print_ids(const std::vector<std::int32_t>& ids) {
  std::printf("[");
  for (std::size_t i = 0; i < ids.size(); ++i) std::printf(i == 0 ? "%d" : ", %d", ids[i]);
  std::printf("]");
}

// Teacher forcing over raw steps. Clip c of a batch uses block-table row c, whose Self blocks
// are c * blocks_per_seq onward.
void feed(AsrEngine& engine, const std::vector<Clip>& clips) {
  const int batch = engine.options().max_encode_batch;
  const int block_size = engine.kv_block_size();
  const int blocks_per_seq = (engine.config().max_ctx + block_size - 1) / block_size;
  std::printf("{\"clips\": [");
  for (std::size_t first = 0; first < clips.size(); first += batch) {
    const std::size_t n = std::min<std::size_t>(batch, clips.size() - first);
    std::vector<float> features;
    std::vector<int> rows(n);
    std::iota(rows.begin(), rows.end(), 0);
    for (std::size_t c = 0; c < n; ++c) {
      const std::vector<float>& own = clips[first + c].features;
      features.insert(features.end(), own.begin(), own.end());
    }
    engine.encode(features, rows);

    for (std::size_t c = 0; c < n; ++c) {
      const Clip& clip = clips[first + c];
      const int row = rows[c];
      std::vector<int> blocks(blocks_per_seq);
      std::iota(blocks.begin(), blocks.end(), row * blocks_per_seq);
      engine.set_blocks(row, blocks);
      const auto step = [&](int start, int count) {
        std::vector<kvb::StepRow> step_rows;
        std::vector<std::int32_t> out_rows;
        for (int i = 0; i < count; ++i) {
          const int pos = start + i;
          step_rows.push_back({.token_id = clip.ids[pos],
                               .position = pos,
                               .seq = row,
                               .kv_slot = row * blocks_per_seq * block_size + pos});
          out_rows.push_back(i);
        }
        engine.forward(step_rows, out_rows);
        const std::span<const std::int32_t> next = engine.next_ids();
        return std::vector<std::int32_t>(next.begin(), next.end());
      };
      const int n_ids = static_cast<int>(clip.ids.size());
      std::printf("%s\n  {\"id\": \"%s\", \"one_step\": ", first + c == 0 ? "" : ",",
                  clip.id.c_str());
      print_ids(step(0, n_ids));
      std::vector<std::int32_t> stepped;
      for (int pos = 0; pos < n_ids; ++pos) stepped.push_back(step(pos, 1)[0]);
      std::printf(", \"token_steps\": ");
      print_ids(stepped);
      std::printf("}");
    }
  }
  std::printf("\n]}\n");
}

int run(int argc, char** argv) {
  Args args(argc, argv);
  const std::string model_dir = args.required("model");
  const std::vector<Clip> clips = load_clips(args.required("clips"));
  const std::string mode = args.str("mode", "feed");

  kvb::cuda::EngineConfig config;
  kvb::cuda::AsrOptions options;
  config.max_ctx = args.integer("max-ctx", 0);
  config.max_seqs = args.integer("slots", 1);
  config.max_step_tokens = args.integer("max-step-tokens", 0);
  config.kv_block_size = args.integer("kv-block-size", 16);
  config.kv_blocks = args.integer("kv-blocks", 0);
  config.use_graph = args.integer("graph", 1) != 0;
  config.max_graphs = args.integer("max-graphs", config.max_graphs);
  config.quantized_gemm_rows = args.integer("quantized-gemm-rows", config.quantized_gemm_rows);
  config.expand_quantized = args.integer("expand-quantized", config.expand_quantized) != 0;
  config.fuse = args.integer("fuse", config.fuse) != 0;
  config.prefill_attention = args.integer("prefill-attention", config.prefill_attention) != 0;
  config.f16_gemv_rows = args.integer("f16-gemv-rows", config.f16_gemv_rows);
  options.max_encode_batch = args.integer("encode-batch", 1);
  const std::string kv_dtype = args.str("kv-dtype", "fp16");
  const std::string cross_kv_dtype = args.str("cross-kv-dtype", "fp16");
  args.expect_consumed();
  const auto self_dtype = kvb::kv_dtype_from_name(kv_dtype);
  const auto cross_dtype = kvb::kv_dtype_from_name(cross_kv_dtype);
  if (!self_dtype || !cross_dtype) throw kvb::Error("unknown --kv-dtype or --cross-kv-dtype");
  config.kv_dtype = *self_dtype;
  options.cross_kv_dtype = *cross_dtype;
  if (mode != "feed") throw kvb::Error("unknown --mode " + mode);
  // Teacher forcing asks for the logits of every row of a step.
  config.max_outputs = std::numeric_limits<int>::max();

  KVB_CUDA(cudaSetDevice(0));
  AsrEngine engine(model_dir, config, options);
  feed(engine, clips);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "kvb-asr: %s\n", e.what());
    return 1;
  }
}
