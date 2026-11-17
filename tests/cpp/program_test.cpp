// The op programs and the activation memory plans built from them, on the fixtures' dimensions.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "kvbound/error.h"
#include "kvbound/json.h"
#include "kvbound/mapped_file.h"
#include "kvbound/memory_plan.h"
#include "kvbound/model_config.h"
#include "kvbound/program.h"
#include "kvbound/safetensors.h"

namespace {

const std::filesystem::path kFixture = KVB_FIXTURE_DIR;
const std::filesystem::path kWhisperFixture = KVB_WHISPER_FIXTURE_DIR;

struct Planned {
  explicit Planned(bool fuse = true) : program(kvb::build_llama_program(config, fuse)) {}
  kvb::ModelConfig config = kvb::load_model_config(kFixture / "config.json");
  kvb::Program program;
  std::vector<kvb::Lifetime> lifetimes = kvb::value_lifetimes(program);
  std::vector<std::size_t> bytes = kvb::value_bytes(program, 48, 8);
};

class FusedOrNot : public testing::TestWithParam<bool> {};

TEST_P(FusedOrNot, ReadsOnlyValuesAlreadyWritten) {
  const Planned p(GetParam());
  std::set<int> written;
  for (const kvb::Op& op : p.program.ops) {
    for (const int in : {op.in0, op.in1}) {
      if (in >= 0) EXPECT_TRUE(written.contains(in)) << p.program.values[in].name;
    }
    if (op.out >= 0) written.insert(op.out);
  }
  EXPECT_EQ(written.size(), p.program.values.size());
}

TEST_P(FusedOrNot, NamesEveryCheckpointTensorAndGoldenSite) {
  const Planned p(GetParam());
  const kvb::SafetensorsFile weights{kFixture / "model.safetensors"};
  std::set<std::string> used;
  for (const kvb::WeightParts& parts : p.program.weights) used.insert(parts.begin(), parts.end());
  for (const std::string& name : used) EXPECT_NE(weights.find(name), nullptr) << name;
  EXPECT_EQ(used.size(), weights.tensors().size());

  std::set<std::string> sites;
  for (const kvb::Op& op : p.program.ops) {
    for (const std::string* site : {&op.site, &op.in0_site}) {
      if (!site->empty()) EXPECT_TRUE(sites.insert(*site).second) << *site;
    }
  }
  const kvb::MappedFile manifest_file{kFixture / "manifest.json"};
  const kvb::json::Value manifest = kvb::json::parse(manifest_file.text());
  const kvb::json::Object& limits =
      manifest.at("thresholds").at("fp16").at("rel_rms_max").as_object();
  for (const auto& [site, limit] : limits) EXPECT_TRUE(sites.contains(site)) << site;
  EXPECT_EQ(sites.size(), limits.size());
}

TEST(Program, FusingTakesNineOpsPerLayer) {
  const Planned fused(true), unfused(false);
  const std::size_t layers = static_cast<std::size_t>(fused.config.n_layers);
  // Outside the layers: embed, gather, norm, logits, argmax, and in the fused program the first
  // layer's input norm.
  EXPECT_EQ(unfused.program.ops.size(), 16 * layers + 5);
  EXPECT_EQ(fused.program.ops.size(), 9 * layers + 6);
}

INSTANTIATE_TEST_SUITE_P(Program, FusedOrNot, testing::Bool());

// The greedy plan of a program: aligned, inside the arena, and no two values that are alive at
// the same time share bytes.
void expect_greedy_plan_is_valid(const kvb::Program& program,
                                 const std::vector<std::size_t>& bytes) {
  const std::vector<kvb::Lifetime> lifetimes = kvb::value_lifetimes(program);
  const kvb::MemoryPlan plan = kvb::plan_memory(bytes, lifetimes, kvb::PlanStrategy::GreedyBySize);
  for (std::size_t a = 0; a < bytes.size(); ++a) {
    EXPECT_EQ(plan.offsets[a] % kvb::kArenaAlign, 0u);
    EXPECT_LE(plan.offsets[a] + bytes[a], plan.arena_bytes);
    for (std::size_t b = a + 1; b < bytes.size(); ++b) {
      const bool alive_together =
          lifetimes[a].first <= lifetimes[b].last && lifetimes[b].first <= lifetimes[a].last;
      const bool share_bytes = plan.offsets[a] < plan.offsets[b] + bytes[b] &&
                               plan.offsets[b] < plan.offsets[a] + bytes[a];
      EXPECT_FALSE(alive_together && share_bytes)
          << program.values[a].name << " and " << program.values[b].name;
    }
  }
}

TEST(MemoryPlan, GreedyNeverOverlapsLiveValues) {
  const Planned p;
  expect_greedy_plan_is_valid(p.program, p.bytes);
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

struct WhisperPlanned {
  explicit WhisperPlanned(bool fuse = true)
      : programs(kvb::build_whisper_programs(config, rules, fuse)) {}
  std::vector<const kvb::Program*> both() const { return {&programs.encoder, &programs.decoder}; }

  kvb::WhisperConfig config = kvb::load_whisper_config(kWhisperFixture / "config.json");
  kvb::WhisperRules rules = kvb::load_whisper_rules(kWhisperFixture / "generation.json");
  kvb::WhisperPrograms programs;
};

// Capture sites in the order a run reports them: the encoder's, then the decoder's.
std::vector<std::string> site_order(const WhisperPlanned& p) {
  std::vector<std::string> sites;
  for (const kvb::Program* program : p.both()) {
    for (const kvb::Op& op : program->ops) {
      for (const std::string* site : {&op.in0_site, &op.site}) {
        if (!site->empty()) sites.push_back(*site);
      }
    }
  }
  return sites;
}

std::vector<const kvb::Op*> ops_of(const kvb::Program& program, kvb::OpKind kind) {
  std::vector<const kvb::Op*> ops;
  for (const kvb::Op& op : program.ops) {
    if (op.kind == kind) ops.push_back(&op);
  }
  return ops;
}

class WhisperFusedOrNot : public testing::TestWithParam<bool> {};

TEST_P(WhisperFusedOrNot, ReadsOnlyValuesAlreadyWritten) {
  const WhisperPlanned p(GetParam());
  for (const kvb::Program* program : p.both()) {
    std::set<int> written;
    // The host fills the input before the first op.
    if (program->input >= 0) written.insert(program->input);
    for (const kvb::Op& op : program->ops) {
      for (const int in : {op.in0, op.in1}) {
        if (in >= 0) EXPECT_TRUE(written.contains(in)) << program->values[in].name;
      }
      if (op.out >= 0) written.insert(op.out);
    }
    EXPECT_EQ(written.size(), program->values.size());
  }
}

TEST_P(WhisperFusedOrNot, SitesAreTheReferenceSitesInForwardOrder) {
  const WhisperPlanned p(GetParam());
  // The manifest lists every site of kvbound.ref.whisper.site_names in order, as the keys of
  // its FP16 thresholds.
  const kvb::MappedFile manifest_file{kWhisperFixture / "manifest.json"};
  const kvb::json::Value manifest = kvb::json::parse(manifest_file.text());
  std::vector<std::string> expected;
  for (const auto& [site, limit] :
       manifest.at("thresholds").at("fp16").at("rel_rms_max").as_object()) {
    expected.push_back(site);
  }
  const std::vector<std::string> sites = site_order(p);
  EXPECT_EQ(sites, expected);
  EXPECT_EQ(std::set<std::string>(sites.begin(), sites.end()).size(), sites.size());

  // One capture map serves both programs: the encoder's sites carry its name.
  for (const kvb::Op& op : p.programs.encoder.ops) {
    for (const std::string* site : {&op.in0_site, &op.site}) {
      EXPECT_TRUE(site->empty() || site->starts_with("encoder.")) << *site;
    }
  }
  for (const kvb::Op& op : p.programs.decoder.ops) {
    for (const std::string* site : {&op.in0_site, &op.site}) {
      EXPECT_TRUE(site->empty() || site->starts_with("decoder.") || *site == "logits") << *site;
    }
  }
}

TEST_P(WhisperFusedOrNot, SharesOneWeightTableNamingEveryCheckpointTensorOnce) {
  const WhisperPlanned p(GetParam());
  const kvb::Program& encoder = p.programs.encoder;
  EXPECT_EQ(encoder.weights, p.programs.decoder.weights);
  EXPECT_EQ(encoder.part_rows, p.programs.decoder.part_rows);
  ASSERT_EQ(encoder.part_rows.size(), encoder.weights.size());

  const kvb::SafetensorsFile weights{kWhisperFixture / "model.safetensors"};
  std::set<std::string> named;
  for (const kvb::WeightParts& parts : encoder.weights) {
    for (const std::string& name : parts) {
      EXPECT_TRUE(named.insert(name).second) << name << " is listed twice";
      EXPECT_NE(weights.find(name), nullptr) << name;
    }
  }
  EXPECT_EQ(named.size(), weights.tensors().size());

  // Every entry of the table is read by an op of one of the programs.
  std::set<int> read;
  for (const kvb::Program* program : p.both()) {
    for (const kvb::Op& op : program->ops) {
      for (const int index : {op.weight, op.bias}) {
        if (index < 0) continue;
        ASSERT_LT(index, static_cast<int>(encoder.weights.size()));
        read.insert(index);
      }
    }
  }
  EXPECT_EQ(read.size(), encoder.weights.size());
}

TEST_P(WhisperFusedOrNot, WeightShapesFitTheValuesTheyMeet) {
  const WhisperPlanned p(GetParam());
  const kvb::SafetensorsFile file{kWhisperFixture / "model.safetensors"};
  for (const kvb::Program* program : p.both()) {
    const bool encoder = program == &p.programs.encoder;
    // Checks the parts of table entry `index` against the rows they must have in total; `cols`
    // is the row length of a matrix, 0 for a one-dimensional tensor.
    const auto check = [&](int index, std::int64_t total_rows, std::int64_t cols) {
      const kvb::WeightParts& parts = program->weights[index];
      const std::vector<std::int64_t>& part_rows = program->part_rows[index];
      ASSERT_TRUE(part_rows.empty() ? parts.size() == 1 : part_rows.size() == parts.size());
      std::int64_t rows = 0;
      for (std::size_t part = 0; part < parts.size(); ++part) {
        SCOPED_TRACE(parts[part]);
        const std::vector<std::int64_t>& shape = file.at(parts[part]).shape;
        const std::int64_t expected = part_rows.empty() ? total_rows : part_rows[part];
        ASSERT_FALSE(shape.empty());
        EXPECT_EQ(shape[0], expected);
        // Trailing dimensions flatten into the columns: a conv weight is [out, in, 3].
        const std::int64_t tail =
            std::accumulate(shape.begin() + 1, shape.end(), std::int64_t{1}, std::multiplies<>());
        EXPECT_EQ(tail, cols == 0 ? 1 : cols);
        EXPECT_EQ(shape.size() == 1, cols == 0);
        rows += expected;
      }
      EXPECT_EQ(rows, total_rows);
    };
    for (const kvb::Op& op : program->ops) {
      SCOPED_TRACE(std::string(kvb::op_name(op.kind)) + " in layer " + std::to_string(op.layer));
      const std::int64_t out_cols = op.out >= 0 ? program->values[op.out].cols : 0;
      switch (op.kind) {
        case kvb::OpKind::Linear:
          check(op.weight, out_cols, program->values[op.in0].cols);
          if (op.bias >= 0) check(op.bias, out_cols, 0);
          break;
        case kvb::OpKind::LayerNorm:
        case kvb::OpKind::AddLayerNorm:
          check(op.weight, out_cols, 0);
          check(op.bias, out_cols, 0);
          break;
        case kvb::OpKind::AddPositions:
          check(op.weight, encoder ? p.config.max_source_positions : p.config.max_target_positions,
                out_cols);
          EXPECT_EQ(op.bias, -1);
          break;
        case kvb::OpKind::Embed:
          check(op.weight, p.config.vocab_size, out_cols);
          EXPECT_EQ(op.bias, -1);
          break;
        default:
          EXPECT_EQ(op.weight, -1);
          EXPECT_EQ(op.bias, -1);
      }
    }
  }
}

TEST_P(WhisperFusedOrNot, EncoderTurnsMelFramesIntoCrossAttentionCaches) {
  const WhisperPlanned p(GetParam());
  const kvb::Program& encoder = p.programs.encoder;
  const std::int64_t d = p.config.d_model;
  const auto value = [&](int id) -> const kvb::Value& { return encoder.values[id]; };

  // The host fills one FP16 row per mel frame. The encoder has no logits.
  ASSERT_GE(encoder.input, 0);
  EXPECT_EQ(value(encoder.input).rows, kvb::RowKind::Frame);
  EXPECT_EQ(value(encoder.input).cols, p.config.n_mels);
  EXPECT_EQ(value(encoder.input).dtype, kvb::DType::F16);
  EXPECT_EQ(encoder.logits, -1);
  EXPECT_EQ(encoder.next_ids, -1);
  EXPECT_EQ(p.config.n_frames(), kvb::kFramesPerToken * p.config.max_source_positions);

  // The first convolution keeps the frame rate; the second, of stride 2, leaves one row per
  // position.
  const std::vector<const kvb::Op*> unfolds = ops_of(encoder, kvb::OpKind::Unfold);
  ASSERT_EQ(unfolds.size(), 2u);
  EXPECT_EQ(&encoder.ops[0], unfolds[0]);
  EXPECT_EQ(unfolds[0]->in0, encoder.input);
  EXPECT_EQ(value(unfolds[0]->out).rows, kvb::RowKind::Frame);
  EXPECT_EQ(value(unfolds[0]->out).cols, 3 * p.config.n_mels);
  EXPECT_EQ(value(unfolds[1]->in0).rows, kvb::RowKind::Frame);
  EXPECT_EQ(value(unfolds[1]->in0).cols, d);
  EXPECT_EQ(value(unfolds[1]->out).rows, kvb::RowKind::Token);
  EXPECT_EQ(value(unfolds[1]->out).cols, 3 * d);

  // Self attention reads q, k and v of the whole clip from one value and caches nothing.
  const std::vector<const kvb::Op*> dense = ops_of(encoder, kvb::OpKind::DenseAttention);
  ASSERT_EQ(dense.size(), static_cast<std::size_t>(p.config.encoder_layers));
  for (const kvb::Op* op : dense) {
    EXPECT_EQ(value(op->in0).cols, 3 * d);
    EXPECT_EQ(value(op->out).cols, d);
  }
  EXPECT_TRUE(ops_of(encoder, kvb::OpKind::Attention).empty());

  // Its only cache writes are k|v of each decoder layer's cross attention, the value captured
  // as encoder.cross_kv.<layer>.
  const std::vector<const kvb::Op*> writes = ops_of(encoder, kvb::OpKind::KvWrite);
  ASSERT_EQ(writes.size(), static_cast<std::size_t>(p.config.decoder_layers));
  for (int j = 0; j < static_cast<int>(writes.size()); ++j) {
    const kvb::Op& write = *writes[j];
    EXPECT_EQ(write.cache, kvb::KvCache::Cross);
    EXPECT_EQ(write.layer, j);
    EXPECT_EQ(write.in1, -1);
    EXPECT_EQ(write.out, -1);
    EXPECT_EQ(value(write.in0).cols, 2 * d);
    const kvb::Op& product = *(&write - 1);
    EXPECT_EQ(product.kind, kvb::OpKind::Linear);
    EXPECT_EQ(product.out, write.in0);
    EXPECT_EQ(product.site, "encoder.cross_kv." + std::to_string(j));
    EXPECT_EQ(encoder.part_rows[product.weight], (std::vector<std::int64_t>{d, d}));
    EXPECT_EQ(encoder.weights[product.bias].size(), 2u);
  }
}

TEST_P(WhisperFusedOrNot, DecoderReadsBothCachesAndMasksItsLogits) {
  const WhisperPlanned p(GetParam());
  const kvb::Program& decoder = p.programs.decoder;
  const std::size_t layers = static_cast<std::size_t>(p.config.decoder_layers);
  EXPECT_EQ(decoder.input, -1);

  // Each layer attends to its own cache, then to the encoder's.
  const std::vector<const kvb::Op*> attention = ops_of(decoder, kvb::OpKind::Attention);
  ASSERT_EQ(attention.size(), 2 * layers);
  for (std::size_t i = 0; i < attention.size(); ++i) {
    EXPECT_EQ(attention[i]->layer, static_cast<int>(i / 2));
    EXPECT_EQ(attention[i]->cache, i % 2 == 0 ? kvb::KvCache::Self : kvb::KvCache::Cross);
  }
  const std::vector<const kvb::Op*> writes = ops_of(decoder, kvb::OpKind::KvWrite);
  ASSERT_EQ(writes.size(), layers);
  for (const kvb::Op* write : writes) {
    EXPECT_EQ(write->cache, kvb::KvCache::Self);
    EXPECT_GE(write->in1, 0);
  }

  // Every projection has a bias except the output projection, which is the token embedding.
  const std::vector<const kvb::Op*> products = ops_of(decoder, kvb::OpKind::Linear);
  for (const kvb::Op* op : products) EXPECT_EQ(op->bias >= 0, op != products.back());
  ASSERT_GE(decoder.ops.size(), 3u);
  const kvb::Op* tail = &decoder.ops[decoder.ops.size() - 3];
  EXPECT_EQ(&tail[0], products.back());
  EXPECT_EQ(tail[0].weight, ops_of(decoder, kvb::OpKind::Embed).at(0)->weight);
  EXPECT_EQ(tail[0].site, "logits");
  EXPECT_EQ(tail[0].out, decoder.logits);
  EXPECT_EQ(decoder.values[decoder.logits].dtype, kvb::DType::F32);

  // The mask sits between the captured logits and the argmax, and carries the decoding rules.
  EXPECT_EQ(tail[1].kind, kvb::OpKind::SuppressTokens);
  EXPECT_EQ(tail[1].in0, decoder.logits);
  EXPECT_EQ(tail[1].out, decoder.logits);
  EXPECT_TRUE(tail[1].site.empty());
  EXPECT_EQ(tail[2].kind, kvb::OpKind::Argmax);
  EXPECT_EQ(tail[2].in0, decoder.logits);
  EXPECT_EQ(tail[2].out, decoder.next_ids);
  EXPECT_EQ(decoder.suppress.always, p.rules.suppress_tokens);
  EXPECT_EQ(decoder.suppress.begin, p.rules.begin_suppress_tokens);
  EXPECT_EQ(decoder.suppress.begin_position, static_cast<int>(p.rules.prompt.size()) - 1);
  EXPECT_TRUE(p.programs.encoder.suppress.always.empty());
}

TEST_P(WhisperFusedOrNot, EachProgramGetsItsOwnMemoryPlan) {
  const WhisperPlanned p(GetParam());
  // One clip per encoder step; decoder steps of up to 48 rows and 8 outputs.
  const std::int64_t n_ctx = p.config.max_source_positions;
  const kvb::Program& encoder = p.programs.encoder;
  const std::vector<std::size_t> encoder_bytes = kvb::value_bytes(encoder, n_ctx, 0);
  expect_greedy_plan_is_valid(encoder, encoder_bytes);
  expect_greedy_plan_is_valid(p.programs.decoder, kvb::value_bytes(p.programs.decoder, 48, 8));

  // A frame value has two rows per position, and the input is alive from the first op.
  const std::size_t mel_bytes =
      static_cast<std::size_t>(2 * n_ctx * p.config.n_mels) * sizeof(std::uint16_t);
  EXPECT_EQ(encoder_bytes[encoder.input],
            (mel_bytes + kvb::kArenaAlign - 1) / kvb::kArenaAlign * kvb::kArenaAlign);
  EXPECT_EQ(kvb::value_lifetimes(encoder)[encoder.input].first, 0);
  EXPECT_EQ(kvb::row_count(kvb::RowKind::Frame, n_ctx, 0), p.config.n_frames());
}

INSTANTIATE_TEST_SUITE_P(WhisperProgram, WhisperFusedOrNot, testing::Bool());

TEST(WhisperProgram, FusingFoldsEveryAddButTheLastIntoANorm) {
  const WhisperPlanned fused(true), unfused(false);
  const std::size_t enc = static_cast<std::size_t>(fused.config.encoder_layers);
  const std::size_t dec = static_cast<std::size_t>(fused.config.decoder_layers);
  const auto count = [](const kvb::Program& program, kvb::OpKind kind) {
    return ops_of(program, kind).size();
  };
  // Encoder: 7 ops of conv stem and positions, 10 per layer, the final norm, and a product and a
  // cache write per decoder layer. Decoder: embedding and positions, 18 per layer, then gather,
  // norm, logits, mask and argmax.
  EXPECT_EQ(unfused.programs.encoder.ops.size(), 7 + 10 * enc + 1 + 2 * dec);
  EXPECT_EQ(unfused.programs.decoder.ops.size(), 2 + 18 * dec + 5);
  for (const kvb::Program* program : unfused.both()) {
    EXPECT_EQ(count(*program, kvb::OpKind::AddLayerNorm), 0u);
  }
  // Fused, every add joins the norm after it. Only the first norm of each program stands
  // alone, and the decoder's last add, whose norm runs on the gathered rows.
  EXPECT_EQ(fused.programs.encoder.ops.size(), 7 + 1 + 8 * enc + 2 * dec);
  EXPECT_EQ(fused.programs.decoder.ops.size(), 2 + 1 + 15 * dec + 5);
  EXPECT_EQ(count(fused.programs.encoder, kvb::OpKind::Add), 0u);
  EXPECT_EQ(count(fused.programs.encoder, kvb::OpKind::AddLayerNorm), 2 * enc);
  EXPECT_EQ(count(fused.programs.encoder, kvb::OpKind::LayerNorm), 1u);
  EXPECT_EQ(count(fused.programs.decoder, kvb::OpKind::Add), 1u);
  EXPECT_EQ(count(fused.programs.decoder, kvb::OpKind::AddLayerNorm), 3 * dec - 1);
  EXPECT_EQ(count(fused.programs.decoder, kvb::OpKind::LayerNorm), 2u);

  // Both forms capture the same sites in the same order and read the same weight table.
  EXPECT_EQ(site_order(fused), site_order(unfused));
  EXPECT_EQ(fused.programs.encoder.weights, unfused.programs.encoder.weights);
}

TEST(WhisperProgram, RejectsRulesThatDoNotFitTheModel) {
  const WhisperPlanned p;
  const auto build = [&](const kvb::WhisperRules& rules) {
    return kvb::build_whisper_programs(p.config, rules);
  };
  kvb::WhisperRules rules = p.rules;
  rules.suppress_tokens.push_back(static_cast<std::int32_t>(p.config.vocab_size));
  EXPECT_THROW(build(rules), kvb::Error);
  rules = p.rules;
  rules.prompt.clear();
  EXPECT_THROW(build(rules), kvb::Error);
  rules = p.rules;
  rules.max_length = p.config.max_target_positions + 1;
  EXPECT_THROW(build(rules), kvb::Error);
  rules.max_length = static_cast<std::int64_t>(rules.prompt.size());
  EXPECT_THROW(build(rules), kvb::Error);
  rules.max_length += 1;
  EXPECT_NO_THROW(build(rules));
}

}  // namespace
