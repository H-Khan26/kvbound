// Runs a trace of generation requests through the engine and prints raw timings as one JSON
// object on stdout. It computes no metrics and writes no files: the Python wrapper
// (kvbound.bench) derives TTFT/TPOT, adds the machine fingerprint and writes the result record.
//
// Trace file: {"requests": [{"id": 0, "prompt": [1, 2, ...], "max_new_tokens": 64}, ...]}

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include "kvbound/cuda/engine.h"
#include "kvbound/cuda/generate.h"
#include "kvbound/json.h"
#include "kvbound/mapped_file.h"

namespace {

using Clock = std::chrono::steady_clock;
using kvb::cuda::Engine;
using kvb::cuda::OpTimes;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Request {
  int id = 0;
  std::vector<std::int32_t> prompt;
  int max_new_tokens = 0;
};

std::vector<Request> load_trace(const std::string& path) {
  const kvb::MappedFile file{path};
  const kvb::json::Value root = kvb::json::parse(file.text());
  std::vector<Request> requests;
  for (const kvb::json::Value& entry : root.at("requests").as_array()) {
    Request request;
    request.id = static_cast<int>(entry.at("id").as_int());
    request.max_new_tokens = static_cast<int>(entry.at("max_new_tokens").as_int());
    for (const kvb::json::Value& token : entry.at("prompt").as_array()) {
      request.prompt.push_back(static_cast<std::int32_t>(token.as_int()));
    }
    requests.push_back(std::move(request));
  }
  if (requests.empty()) throw kvb::Error("trace has no requests");
  return requests;
}

// --name value pairs.
class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string key = argv[i];
      if (key.rfind("--", 0) != 0) throw kvb::Error("expected --option, got " + key);
      values_[key.substr(2)] = argv[i + 1];
    }
    if (argc % 2 == 0) throw kvb::Error("every option needs a value");
  }
  std::string str(const std::string& key, const std::string& fallback = {}) {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      if (fallback.empty()) throw kvb::Error("missing --" + key);
      return fallback;
    }
    const std::string value = it->second;
    values_.erase(it);
    return value;
  }
  int integer(const std::string& key, int fallback) {
    return std::atoi(str(key, std::to_string(fallback)).c_str());
  }
  void expect_consumed() const {
    if (!values_.empty()) throw kvb::Error("unknown option --" + values_.begin()->first);
  }

 private:
  std::map<std::string, std::string> values_;
};

template <typename T, typename Format>
void print_array(const char* name, const std::vector<T>& values, Format format) {
  std::printf("\"%s\": [", name);
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::printf(i == 0 ? "" : ", ");
    format(values[i]);
  }
  std::printf("]");
}

void print_op_times(const char* name, const OpTimes& times, int steps) {
  std::printf("    \"%s\": {\"steps\": %d, \"ops\": {", name, steps);
  bool first = true;
  for (int kind = 0; kind < kvb::kOpKindCount; ++kind) {
    if (times.calls[kind] == 0) continue;
    std::printf("%s\"%s\": {\"calls\": %lld, \"seconds\": %.9f}", first ? "" : ", ",
                std::string(kvb::op_name(static_cast<kvb::OpKind>(kind))).c_str(),
                static_cast<long long>(times.calls[kind]), times.seconds[kind]);
    first = false;
  }
  std::printf("}}");
}

// One request alone on the engine. Token times are seconds from submission.
struct Timed {
  std::vector<double> token_times;
  std::vector<std::int32_t> tokens;
};

Timed run_request(Engine& engine, const Request& request) {
  Timed timed;
  const Clock::time_point start = Clock::now();
  timed.tokens = kvb::cuda::generate(
      engine, request.prompt,
      {.max_new_tokens = request.max_new_tokens,
       .on_token = [&](std::int32_t) { timed.token_times.push_back(seconds_since(start)); }});
  return timed;
}

int run(int argc, char** argv) {
  Args args(argc, argv);
  const std::string model_dir = args.str("model");
  const std::vector<Request> requests = load_trace(args.str("trace"));
  const int warmup = args.integer("warmup", 1);
  const int iters = args.integer("iters", 3);
  const bool profile_ops = args.integer("profile-ops", 0) != 0;

  kvb::cuda::EngineConfig config;
  config.max_ctx = args.integer("max-ctx", 2048);
  config.max_step_tokens = args.integer("max-step-tokens", 0);
  config.use_graph = args.integer("graph", 1) != 0;
  const std::string layout = args.str("kv-layout", "head_major");
  const std::string gemm = args.str("gemm", "fp32");
  const std::string plan = args.str("plan", "greedy");
  args.expect_consumed();
  if (const auto value = kvb::kv_layout_from_name(layout)) {
    config.kv_layout = *value;
  } else {
    throw kvb::Error("unknown --kv-layout " + layout);
  }
  if (const auto value = kvb::cuda::gemm_accumulate_from_name(gemm)) {
    config.gemm = *value;
  } else {
    throw kvb::Error("unknown --gemm " + gemm);
  }
  if (plan != "greedy" && plan != "naive") throw kvb::Error("unknown --plan " + plan);
  config.plan = plan == "naive" ? kvb::PlanStrategy::Naive : kvb::PlanStrategy::GreedyBySize;

  KVB_CUDA(cudaSetDevice(0));
  cudaDeviceProp prop{};
  KVB_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::size_t free_before = 0, free_after = 0, total = 0;
  KVB_CUDA(cudaMemGetInfo(&free_before, &total));
  const Clock::time_point load_start = Clock::now();
  Engine engine(model_dir, config);
  const double load_s = seconds_since(load_start);
  KVB_CUDA(cudaMemGetInfo(&free_after, &total));

  for (int i = 0; i < warmup; ++i) {
    for (const Request& request : requests) run_request(engine, request);
  }
  std::vector<std::vector<Timed>> iterations;
  for (int i = 0; i < iters; ++i) {
    std::vector<Timed>& runs = iterations.emplace_back();
    for (const Request& request : requests) runs.push_back(run_request(engine, request));
  }

  // One more pass with every op synchronized and timed, split at the first token.
  OpTimes prefill, decode;
  int decode_steps = 0;
  if (profile_ops) {
    for (const Request& request : requests) {
      OpTimes times;
      engine.set_op_times(&times);
      bool first = true;
      kvb::cuda::generate(engine, request.prompt,
                          {.max_new_tokens = request.max_new_tokens, .on_token = [&](std::int32_t) {
                             OpTimes& into = first ? prefill : decode;
                             for (int kind = 0; kind < kvb::kOpKindCount; ++kind) {
                               into.seconds[kind] += times.seconds[kind];
                               into.calls[kind] += times.calls[kind];
                             }
                             decode_steps += first ? 0 : 1;
                             first = false;
                             times = OpTimes{};
                           }});
      engine.set_op_times(nullptr);
    }
  }

  const kvb::ModelConfig& model = engine.model_config();
  const kvb::cuda::EngineConfig& resolved = engine.config();
  const kvb::cuda::EngineMemory memory = engine.memory();
  std::printf("{\n");
  std::printf(
      "  \"device\": {\"name\": \"%s\", \"compute_capability\": \"%d.%d\", "
      "\"total_memory_bytes\": %zu},\n",
      prop.name, prop.major, prop.minor, prop.totalGlobalMem);
  std::printf(
      "  \"config\": {\"max_ctx\": %d, \"max_seqs\": %d, \"max_step_tokens\": %d, "
      "\"kv_block_size\": %d, \"kv_blocks\": %d, \"kv_layout\": \"%s\", "
      "\"gemm_accumulate\": \"%s\", \"graph\": %s, \"plan\": \"%s\", \"warmup\": %d, "
      "\"iters\": %d},\n",
      resolved.max_ctx, resolved.max_seqs, resolved.max_step_tokens, resolved.kv_block_size,
      resolved.kv_blocks, layout.c_str(), gemm.c_str(), resolved.use_graph ? "true" : "false",
      plan.c_str(), warmup, iters);
  std::printf(
      "  \"model\": {\"vocab_size\": %lld, \"hidden_size\": %lld, \"n_layers\": %lld, "
      "\"n_heads\": %lld, \"n_kv_heads\": %lld, \"intermediate_size\": %lld},\n",
      static_cast<long long>(model.vocab_size), static_cast<long long>(model.hidden_size),
      static_cast<long long>(model.n_layers), static_cast<long long>(model.n_heads),
      static_cast<long long>(model.n_kv_heads), static_cast<long long>(model.intermediate_size));
  std::printf("  \"program\": {\"ops\": %zu, \"values\": %zu},\n", engine.program().ops.size(),
              engine.program().values.size());
  std::printf(
      "  \"memory\": {\"weights_bytes\": %zu, \"kv_pool_bytes\": %zu, \"arena_bytes\": %zu, "
      "\"arena_naive_bytes\": %zu, \"arena_peak_live_bytes\": %zu, \"device_bytes\": %zu},\n",
      memory.weights, memory.kv_pool, memory.arena, memory.arena_naive, memory.arena_peak_live,
      free_before - free_after);
  std::printf("  \"load_s\": %.6f,\n", load_s);
  std::printf("  \"iterations\": [\n");
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    std::printf("    {\"requests\": [\n");
    for (std::size_t r = 0; r < requests.size(); ++r) {
      const Timed& timed = iterations[i][r];
      std::printf("      {\"id\": %d, \"n_prompt\": %zu, ", requests[r].id,
                  requests[r].prompt.size());
      print_array("token_times_s", timed.token_times, [](double t) { std::printf("%.9f", t); });
      std::printf(", ");
      print_array("tokens", timed.tokens, [](std::int32_t t) { std::printf("%d", t); });
      std::printf("}%s\n", r + 1 < requests.size() ? "," : "");
    }
    std::printf("    ]}%s\n", i + 1 < iterations.size() ? "," : "");
  }
  std::printf("  ]");
  if (profile_ops) {
    std::printf(",\n  \"op_profile\": {\n");
    print_op_times("prefill", prefill, static_cast<int>(requests.size()));
    std::printf(",\n");
    print_op_times("decode", decode, decode_steps);
    std::printf("\n  }");
  }
  std::printf("\n}\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "kvb-bench: %s\n", e.what());
    return 1;
  }
}
