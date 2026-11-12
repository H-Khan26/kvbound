// The op program and the activation memory plan built from it, on the fixture's dimensions.

#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "kvbound/json.h"
#include "kvbound/mapped_file.h"
#include "kvbound/memory_plan.h"
#include "kvbound/model_config.h"
#include "kvbound/program.h"
#include "kvbound/safetensors.h"

namespace {

const std::filesystem::path kFixture = KVB_FIXTURE_DIR;

struct Planned {
  kvb::ModelConfig config = kvb::load_model_config(kFixture / "config.json");
  kvb::Program program = kvb::build_llama_program(config);
  std::vector<kvb::Lifetime> lifetimes = kvb::value_lifetimes(program);
  std::vector<std::size_t> bytes = kvb::value_bytes(program, 48, 8);
};

TEST(Program, ReadsOnlyValuesAlreadyWritten) {
  const Planned p;
  std::set<int> written;
  for (const kvb::Op& op : p.program.ops) {
    for (const int in : {op.in0, op.in1}) {
      if (in >= 0) EXPECT_TRUE(written.contains(in)) << p.program.values[in].name;
    }
    if (op.out >= 0) written.insert(op.out);
  }
  EXPECT_EQ(written.size(), p.program.values.size());
}

TEST(Program, NamesEveryCheckpointTensorAndGoldenSite) {
  const Planned p;
  const kvb::SafetensorsFile weights{kFixture / "model.safetensors"};
  const std::set<std::string> used(p.program.weights.begin(), p.program.weights.end());
  for (const std::string& name : used) EXPECT_NE(weights.find(name), nullptr) << name;
  EXPECT_EQ(used.size(), weights.tensors().size());

  std::set<std::string> sites;
  for (const kvb::Op& op : p.program.ops) {
    if (!op.site.empty()) EXPECT_TRUE(sites.insert(op.site).second) << op.site;
  }
  const kvb::MappedFile manifest_file{kFixture / "manifest.json"};
  const kvb::json::Value manifest = kvb::json::parse(manifest_file.text());
  const kvb::json::Object& limits =
      manifest.at("thresholds").at("fp16").at("rel_rms_max").as_object();
  for (const auto& [site, limit] : limits) EXPECT_TRUE(sites.contains(site)) << site;
  EXPECT_EQ(sites.size(), limits.size());
}

TEST(MemoryPlan, GreedyNeverOverlapsLiveValues) {
  const Planned p;
  const kvb::MemoryPlan plan =
      kvb::plan_memory(p.bytes, p.lifetimes, kvb::PlanStrategy::GreedyBySize);
  for (std::size_t a = 0; a < p.bytes.size(); ++a) {
    EXPECT_EQ(plan.offsets[a] % kvb::kArenaAlign, 0u);
    EXPECT_LE(plan.offsets[a] + p.bytes[a], plan.arena_bytes);
    for (std::size_t b = a + 1; b < p.bytes.size(); ++b) {
      const bool alive_together = p.lifetimes[a].first <= p.lifetimes[b].last &&
                                  p.lifetimes[b].first <= p.lifetimes[a].last;
      const bool share_bytes = plan.offsets[a] < plan.offsets[b] + p.bytes[b] &&
                               plan.offsets[b] < plan.offsets[a] + p.bytes[a];
      EXPECT_FALSE(alive_together && share_bytes)
          << p.program.values[a].name << " and " << p.program.values[b].name;
    }
  }
}

TEST(MemoryPlan, GreedySitsBetweenTheLowerBoundAndNaive) {
  const Planned p;
  const std::size_t naive =
      kvb::plan_memory(p.bytes, p.lifetimes, kvb::PlanStrategy::Naive).arena_bytes;
  const std::size_t greedy =
      kvb::plan_memory(p.bytes, p.lifetimes, kvb::PlanStrategy::GreedyBySize).arena_bytes;
  EXPECT_EQ(naive, std::accumulate(p.bytes.begin(), p.bytes.end(), std::size_t{0}));
  EXPECT_GE(greedy, kvb::peak_live_bytes(p.bytes, p.lifetimes));
  // Two layers of intermediates collapse to roughly one layer's worth.
  EXPECT_LT(greedy, naive / 2);
}

TEST(MemoryPlan, ResidualStreamLivesThroughEveryLayer) {
  const Planned p;
  // Value 0 is the residual stream: written by the embedding, last read by the output gather.
  EXPECT_EQ(p.program.values[0].name, "h");
  EXPECT_EQ(p.lifetimes[0].first, 0);
  EXPECT_EQ(p.program.ops[p.lifetimes[0].last].kind, kvb::OpKind::GatherRows);
  // The outputs stay valid after the last op, when the host reads them.
  EXPECT_EQ(p.lifetimes[p.program.logits].last, static_cast<int>(p.program.ops.size()) - 1);
  EXPECT_EQ(p.lifetimes[p.program.next_ids].last, static_cast<int>(p.program.ops.size()) - 1);
}

}  // namespace
