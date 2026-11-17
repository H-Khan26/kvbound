// Runs a trace of generation requests through the scheduler and the engine and prints raw
// timings as one JSON object on stdout. It computes no metrics and writes no files: the Python
// wrapper (kvbound.bench) derives TTFT/TPOT and memory use, adds the machine fingerprint and
// writes the result record.
//
// A request is queued at its arrival time, zero unless the trace says otherwise. --policy picks
// how the --slots sequences that fit in the cache are chosen: static batches, or continuous
// batching, which is meant to run with small KV blocks (--kv-block-size, --kv-blocks).
//
// Trace file: {"requests": [{"id": 0, "prompt": [1, 2, ...], "max_new_tokens": 64,
//                            "arrival_s": 0.0}, ...]}

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "kvbound/cuda/engine.h"
#include "kvbound/cuda/serve.h"
#include "kvbound/json.h"
#include "kvbound/mapped_file.h"

namespace {

using Clock = std::chrono::steady_clock;
using kvb::cuda::Engine;
using kvb::cuda::OpTimes;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

using Request = kvb::cuda::ServeRequest;

std::vector<Request> load_trace(const std::string& path) {
  const kvb::MappedFile file{path};
  const kvb::json::Value root = kvb::json::parse(file.text());
  std::vector<Request> requests;
  for (const kvb::json::Value& entry : root.at("requests").as_array()) {
    Request request;
    request.id = static_cast<int>(entry.at("id").as_int());
    request.max_new_tokens = static_cast<int>(entry.at("max_new_tokens").as_int());
    if (const kvb::json::Value* arrival = entry.find("arrival_s")) {
      request.arrival_s = arrival->as_number();
    }
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
    if (argc % 2 == 0) throw kvb::Error("every option needs a value");
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string key = argv[i];
      if (key.rfind("--", 0) != 0) throw kvb::Error("expected --option, got " + key);
      values_[key.substr(2)] = argv[i + 1];
    }
  }
  std::string required(const std::string& key) {
    const auto it = values_.find(key);
    if (it == values_.end()) throw kvb::Error("missing --" + key);
    return take(it);
  }
  std::string str(const std::string& key, const std::string& fallback) {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : take(it);
  }
  int integer(const std::string& key, int fallback) {
    return std::atoi(str(key, std::to_string(fallback)).c_str());
  }
  void expect_consumed() const {
    if (!values_.empty()) throw kvb::Error("unknown option --" + values_.begin()->first);
  }

 private:
  std::string take(std::map<std::string, std::string>::iterator it) {
    std::string value = std::move(it->second);
    values_.erase(it);
    return value;
  }
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

// Synchronized per-op times, split by whether a step processed any prompt tokens.
struct OpProfile {
  OpTimes prefill, decode;
  int prefill_steps = 0, decode_steps = 0;
};

// One pass over the trace. Times are seconds from the start of the pass.
struct Pass {
  double total_s = 0.0;
  // Per step: when it was planned and when it finished, the rows it processed and the sequences
  // they belong to, the sequences it preempted, and the queue and cache occupancy after it.
  std::vector<double> step_start_s, step_end_s;
  std::vector<int> step_rows, step_seqs, step_preempted, step_running, step_waiting;
  std::vector<long long> step_reserved, step_used;
  kvb::cuda::GraphStats graphs;  // of this pass alone
  // Per request, in trace order.
  std::vector<std::vector<double>> token_times;
  std::vector<std::vector<std::int32_t>> tokens;
};

Pass run_trace(Engine& engine, const std::vector<Request>& requests,
               const kvb::cuda::ServeOptions& options, OpProfile* profile) {
  std::map<int, std::size_t> index;
  for (std::size_t i = 0; i < requests.size(); ++i) index[requests[i].id] = i;

  Pass pass;
  pass.token_times.resize(requests.size());
  pass.tokens.resize(requests.size());
  OpTimes times;
  std::set<int> emitted;  // sequences that have produced a token: they are decoding
  if (profile != nullptr) engine.set_op_times(&times);
  const kvb::cuda::GraphStats before = engine.graph_stats();
  const Clock::time_point start = Clock::now();
  kvb::cuda::serve(engine, requests, options, [&](const kvb::cuda::ServeStep& step) {
    const double now = step.time_s;
    std::vector<int> emitted_now;
    int rows = 0;
    for (const kvb::StepPlan::Run& run : step.plan.run) rows += run.n_tokens;
    for (const kvb::StepResult::Token& token : step.result.tokens) {
      emitted_now.push_back(token.seq);
      pass.token_times[index.at(token.seq)].push_back(now);
      pass.tokens[index.at(token.seq)].push_back(token.token);
    }
    pass.step_start_s.push_back(step.start_s);
    pass.step_end_s.push_back(now);
    pass.step_rows.push_back(rows);
    pass.step_seqs.push_back(static_cast<int>(step.plan.run.size()));
    pass.step_preempted.push_back(static_cast<int>(step.plan.preempt.size()));
    pass.step_running.push_back(step.usage.running);
    pass.step_waiting.push_back(step.usage.waiting);
    pass.step_reserved.push_back(step.usage.reserved);
    pass.step_used.push_back(step.usage.used);

    if (profile != nullptr) {
      bool prefill = false;
      for (const kvb::StepPlan::Run& run : step.plan.run) {
        prefill = prefill || !emitted.contains(run.seq);
      }
      OpTimes& into = prefill ? profile->prefill : profile->decode;
      (prefill ? profile->prefill_steps : profile->decode_steps) += 1;
      for (int kind = 0; kind < kvb::kOpKindCount; ++kind) {
        into.seconds[kind] += times.seconds[kind];
        into.calls[kind] += times.calls[kind];
      }
      times = OpTimes{};
    }
    emitted.insert(emitted_now.begin(), emitted_now.end());
  });
  pass.total_s = seconds_since(start);
  engine.set_op_times(nullptr);
  const kvb::cuda::GraphStats& after = engine.graph_stats();
  pass.graphs = {.replayed = after.replayed - before.replayed,
                 .eager = after.eager - before.eager,
                 .captured = after.captured - before.captured,
                 .evicted = after.evicted - before.evicted,
                 .capture_s = after.capture_s - before.capture_s};
  return pass;
}

void print_pass(const Pass& pass, const std::vector<Request>& requests, bool last) {
  std::printf("    {\"total_s\": %.9f,\n", pass.total_s);
  std::printf(
      "     \"graphs\": {\"replayed_steps\": %lld, \"eager_steps\": %lld, \"captured\": %d, "
      "\"evicted\": %d, \"capture_s\": %.6f},\n     \"steps\": {",
      static_cast<long long>(pass.graphs.replayed), static_cast<long long>(pass.graphs.eager),
      pass.graphs.captured, pass.graphs.evicted, pass.graphs.capture_s);
  const auto print_int = [](int n) { std::printf("%d", n); };
  print_array("start_s", pass.step_start_s, [](double t) { std::printf("%.6f", t); });
  std::printf(", ");
  print_array("end_s", pass.step_end_s, [](double t) { std::printf("%.6f", t); });
  std::printf(", ");
  print_array("rows", pass.step_rows, print_int);
  std::printf(", ");
  print_array("seqs", pass.step_seqs, print_int);
  std::printf(", ");
  print_array("preempted", pass.step_preempted, print_int);
  std::printf(", ");
  print_array("running", pass.step_running, print_int);
  std::printf(", ");
  print_array("waiting", pass.step_waiting, print_int);
  std::printf(", ");
  print_array("reserved_tokens", pass.step_reserved, [](long long n) { std::printf("%lld", n); });
  std::printf(", ");
  print_array("used_tokens", pass.step_used, [](long long n) { std::printf("%lld", n); });
  std::printf("},\n     \"requests\": [\n");
  for (std::size_t r = 0; r < requests.size(); ++r) {
    std::printf("      {\"id\": %d, \"n_prompt\": %zu, \"arrival_s\": %.6f, ", requests[r].id,
                requests[r].prompt.size(), requests[r].arrival_s);
    print_array("token_times_s", pass.token_times[r], [](double t) { std::printf("%.9f", t); });
    std::printf(", ");
    print_array("tokens", pass.tokens[r], [](std::int32_t t) { std::printf("%d", t); });
    std::printf("}%s\n", r + 1 < requests.size() ? "," : "");
  }
  std::printf("    ]}%s\n", last ? "" : ",");
}

int run(int argc, char** argv) {
  Args args(argc, argv);
  const std::string model_dir = args.required("model");
  const std::vector<Request> requests = load_trace(args.required("trace"));
  const int warmup = args.integer("warmup", 1);
  const int iters = args.integer("iters", 3);
  const bool profile_ops = args.integer("profile-ops", 0) != 0;

  kvb::cuda::EngineConfig config;
  config.max_seqs = args.integer("slots", 1);
  config.max_ctx = args.integer("max-ctx", 2048);
  config.max_step_tokens = args.integer("max-step-tokens", 0);
  config.kv_block_size = args.integer("kv-block-size", 0);
  config.kv_blocks = args.integer("kv-blocks", 0);
  config.use_graph = args.integer("graph", 1) != 0;
  config.max_graphs = args.integer("max-graphs", config.max_graphs);
  const std::string policy = args.str("policy", "static");
  config.quantized_gemm_rows = args.integer("quantized-gemm-rows", config.quantized_gemm_rows);
  config.fuse = args.integer("fuse", 1) != 0;
  config.f16_gemv_rows = args.integer("f16-gemv-rows", config.f16_gemv_rows);
  const std::string kv_dtype = args.str("kv-dtype", "fp16");
  if (const auto value = kvb::kv_dtype_from_name(kv_dtype)) {
    config.kv_dtype = *value;
  } else {
    throw kvb::Error("unknown --kv-dtype " + kv_dtype);
  }
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
  kvb::cuda::ServeOptions serve_options;
  if (const auto value = kvb::schedule_policy_from_name(policy)) {
    serve_options.policy = *value;
  } else {
    throw kvb::Error("unknown --policy " + policy);
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

  const kvb::cuda::EngineConfig& resolved = engine.config();
  for (int i = 0; i < warmup; ++i) run_trace(engine, requests, serve_options, nullptr);
  std::vector<Pass> passes;
  for (int i = 0; i < iters; ++i) {
    passes.push_back(run_trace(engine, requests, serve_options, nullptr));
  }
  // One more pass with every op synchronized and timed.
  OpProfile profile;
  if (profile_ops) run_trace(engine, requests, serve_options, &profile);

  const kvb::ModelConfig& model = engine.model_config();
  const kvb::cuda::EngineMemory memory = engine.memory();
  std::printf("{\n");
  std::printf(
      "  \"device\": {\"name\": \"%s\", \"compute_capability\": \"%d.%d\", "
      "\"total_memory_bytes\": %zu},\n",
      prop.name, prop.major, prop.minor, prop.totalGlobalMem);
  std::printf(
      "  \"config\": {\"policy\": \"%s\", \"slots\": %d, \"max_ctx\": %d, "
      "\"max_step_tokens\": %d, \"max_outputs\": %d, "
      "\"kv_block_size\": %d, \"kv_blocks\": %d, \"kv_layout\": \"%s\", \"kv_dtype\": \"%s\", "
      "\"weights\": \"%s\", \"weight_group_size\": %d, \"quantized_gemm_rows\": %d, "
      "\"gemm_accumulate\": \"%s\", \"graph\": %s, \"max_graphs\": %d, \"plan\": \"%s\", "
      "\"fuse\": %s, \"f16_gemv_rows\": %d, \"warmup\": %d, \"iters\": %d},\n",
      policy.c_str(), resolved.max_seqs, resolved.max_ctx, resolved.max_step_tokens,
      resolved.max_outputs, resolved.kv_block_size, resolved.kv_blocks, layout.c_str(),
      kv_dtype.c_str(), std::string(kvb::weight_scheme_name(engine.weight_format().scheme)).c_str(),
      engine.weight_format().group_size, resolved.quantized_gemm_rows, gemm.c_str(),
      resolved.use_graph ? "true" : "false", resolved.max_graphs, plan.c_str(),
      resolved.fuse ? "true" : "false", resolved.f16_gemv_rows, warmup, iters);
  std::printf(
      "  \"model\": {\"vocab_size\": %lld, \"hidden_size\": %lld, \"n_layers\": %lld, "
      "\"n_heads\": %lld, \"n_kv_heads\": %lld, \"intermediate_size\": %lld},\n",
      static_cast<long long>(model.vocab_size), static_cast<long long>(model.hidden_size),
      static_cast<long long>(model.n_layers), static_cast<long long>(model.n_heads),
      static_cast<long long>(model.n_kv_heads), static_cast<long long>(model.intermediate_size));
  std::printf("  \"program\": {\"ops\": %zu, \"values\": %zu},\n", engine.program().ops.size(),
              engine.program().values.size());
  std::printf(
      "  \"memory\": {\"weights_bytes\": %zu, \"kv_pool_bytes\": %zu, "
      "\"kv_bytes_per_token\": %zu, \"arena_bytes\": %zu, \"arena_naive_bytes\": %zu, "
      "\"arena_peak_live_bytes\": %zu, \"dequant_bytes\": %zu, \"device_bytes\": %zu},\n",
      memory.weights, memory.kv_pool,
      kvb::kv_bytes_per_token(model.n_layers, model.n_kv_heads, model.head_dim(),
                              resolved.kv_dtype),
      memory.arena, memory.arena_naive, memory.arena_peak_live, memory.dequant,
      free_before - free_after);
  std::printf("  \"load_s\": %.6f,\n", load_s);
  // The activation plan: one entry per program value, in the order values are first written.
  const std::vector<kvb::Lifetime> lifetimes = kvb::value_lifetimes(engine.program());
  std::printf("  \"plan\": [\n");
  for (std::size_t v = 0; v < lifetimes.size(); ++v) {
    std::printf(
        "    {\"name\": \"%s\", \"bytes\": %zu, \"offset\": %zu, \"first_op\": %d, "
        "\"last_op\": %d}%s\n",
        engine.program().values[v].name.c_str(), engine.value_sizes()[v],
        engine.memory_plan().offsets[v], lifetimes[v].first, lifetimes[v].last,
        v + 1 < lifetimes.size() ? "," : "");
  }
  std::printf("  ],\n");
  std::printf("  \"iterations\": [\n");
  for (std::size_t i = 0; i < passes.size(); ++i) {
    print_pass(passes[i], requests, i + 1 == passes.size());
  }
  std::printf("  ]");
  if (profile_ops) {
    std::printf(",\n  \"op_profile\": {\n");
    print_op_times("prefill", profile.prefill, profile.prefill_steps);
    std::printf(",\n");
    print_op_times("decode", profile.decode, profile.decode_steps);
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
