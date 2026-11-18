// The Python-to-C++ contract of the Whisper fixture: the config and decoding-rules loaders, the
// layout of a converted checkpoint, and the op programs themselves. A CPU executor runs the
// encoder and decoder programs as kvbound/program.h defines their ops and must reproduce every
// golden site within the manifest's FP32 thresholds and every golden greedy transcription.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "kvbound/dtype.h"
#include "kvbound/error.h"
#include "kvbound/json.h"
#include "kvbound/mapped_file.h"
#include "kvbound/model_config.h"
#include "kvbound/program.h"
#include "kvbound/safetensors.h"
#include "kvbound/step.h"
#include "support/tolerance.h"
#include "support/whisper_cpu.h"

namespace {

using kvb::StepRow;
using kvb::test::Capture;
using kvb::test::CpuWhisper;

const std::filesystem::path kFixture = KVB_WHISPER_FIXTURE_DIR;

struct Fixture {
  explicit Fixture(bool fuse = false)
      : programs(kvb::build_whisper_programs(config, rules, fuse)) {}

  // Features of a golden prompt with one row per frame: the file stores [n_mels, frames].
  std::vector<float> mel(const std::string& prompt) const {
    const auto stored = goldens.at(prompt + "/input_features").as<float>();
    const std::size_t n_mels = static_cast<std::size_t>(config.n_mels);
    const std::size_t n_frames = static_cast<std::size_t>(config.n_frames());
    EXPECT_EQ(stored.size(), n_mels * n_frames);
    std::vector<float> rows(stored.size());
    for (std::size_t m = 0; m < n_mels; ++m) {
      for (std::size_t f = 0; f < n_frames; ++f) rows[f * n_mels + m] = stored[m * n_frames + f];
    }
    return rows;
  }
  // Step rows of the encoder pass over clip number `clip` of a step, into cross block `block`.
  std::vector<StepRow> encoder_rows(int clip, int block) const {
    const std::int32_t n_ctx = static_cast<std::int32_t>(config.max_source_positions);
    std::vector<StepRow> rows;
    for (std::int32_t p = 0; p < n_ctx; ++p) rows.push_back({0, p, clip, block * n_ctx + p});
    return rows;
  }
  // Step rows of tokens at consecutive positions from `first`, of block-table row `seq`.
  std::vector<StepRow> decoder_rows(std::span<const std::int32_t> ids, int seq,
                                    std::int32_t first = 0) const {
    const std::int32_t max_ctx = static_cast<std::int32_t>(config.max_target_positions);
    std::vector<StepRow> rows;
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(ids.size()); ++i) {
      rows.push_back({ids[i], first + i, seq, seq * max_ctx + first + i});
    }
    return rows;
  }
  // The golden prompt ids, in manifest order.
  std::vector<std::string> prompts() const {
    std::vector<std::string> ids;
    for (const kvb::json::Value& entry : manifest.at("prompts").as_array()) {
      ids.push_back(entry.at("id").as_string());
    }
    return ids;
  }
  // Checks every captured site of the listed clips, whose rows sit one clip after another,
  // against the goldens.
  void expect_goldens(const Capture& capture, const std::vector<std::string>& clips) const {
    const kvb::json::Value& fp32 = manifest.at("thresholds").at("fp32");
    for (const auto& [site, limit] :
         manifest.at("thresholds").at("fp16").at("rel_rms_max").as_object()) {
      ASSERT_TRUE(capture.contains(site)) << site;
      std::vector<float> expected;
      for (const std::string& clip : clips) {
        const auto golden = goldens.at(clip + "/" + site).as<float>();
        expected.insert(expected.end(), golden.begin(), golden.end());
      }
      const std::vector<float>& actual = capture.at(site);
      ASSERT_EQ(actual.size(), expected.size()) << site;
      EXPECT_LE(kvb::test::rel_rms(actual, expected), fp32.at("rel_rms_max").as_number()) << site;
      EXPECT_GE(kvb::test::cosine(actual, expected), fp32.at("cosine_min").as_number()) << site;
    }
  }

  kvb::WhisperConfig config = kvb::load_whisper_config(kFixture / "config.json");
  kvb::WhisperRules rules = kvb::load_whisper_rules(kFixture / "generation.json");
  kvb::WhisperPrograms programs;
  kvb::SafetensorsFile weights{kFixture / "model.safetensors"};
  kvb::SafetensorsFile goldens{kFixture / "goldens.safetensors"};
  kvb::MappedFile manifest_file{kFixture / "manifest.json"};
  kvb::json::Value manifest = kvb::json::parse(manifest_file.text());
};

std::vector<std::int32_t> token_ids(const kvb::json::Value& list) {
  std::vector<std::int32_t> ids;
  for (const kvb::json::Value& id : list.as_array()) {
    ids.push_back(static_cast<std::int32_t>(id.as_int()));
  }
  return ids;
}

TEST(WhisperContract, LoadersAgreeWithThePythonReference) {
  const Fixture fx;
  const kvb::WhisperConfig& cfg = fx.config;
  // The manifest holds what kvbound.ref.WhisperConfig read from the same config.json.
  const kvb::json::Value& ref = fx.manifest.at("model").at("config");
  EXPECT_EQ(ref.as_object().size(), 12u);
  EXPECT_EQ(cfg.vocab_size, ref.at("vocab_size").as_int());
  EXPECT_EQ(cfg.d_model, ref.at("d_model").as_int());
  EXPECT_EQ(cfg.n_mels, ref.at("n_mels").as_int());
  EXPECT_EQ(cfg.encoder_layers, ref.at("encoder_layers").as_int());
  EXPECT_EQ(cfg.encoder_heads, ref.at("encoder_heads").as_int());
  EXPECT_EQ(cfg.encoder_ffn_dim, ref.at("encoder_ffn_dim").as_int());
  EXPECT_EQ(cfg.decoder_layers, ref.at("decoder_layers").as_int());
  EXPECT_EQ(cfg.decoder_heads, ref.at("decoder_heads").as_int());
  EXPECT_EQ(cfg.decoder_ffn_dim, ref.at("decoder_ffn_dim").as_int());
  EXPECT_EQ(cfg.max_source_positions, ref.at("max_source_positions").as_int());
  EXPECT_EQ(cfg.max_target_positions, ref.at("max_target_positions").as_int());
  EXPECT_EQ(cfg.layer_norm_eps, ref.at("layer_norm_eps").as_number());

  // generation.json states the rules that the fixture's Hugging Face config implies.
  const kvb::MappedFile config_file{kFixture / "config.json"};
  const kvb::json::Value hf = kvb::json::parse(config_file.text());
  std::vector<std::int32_t> prompt{
      static_cast<std::int32_t>(hf.at("decoder_start_token_id").as_int())};
  for (const kvb::json::Value& forced : hf.at("forced_decoder_ids").as_array()) {
    prompt.push_back(static_cast<std::int32_t>(forced.as_array().at(1).as_int()));
  }
  EXPECT_EQ(fx.rules.prompt, prompt);
  EXPECT_EQ(fx.rules.suppress_tokens, token_ids(hf.at("suppress_tokens")));
  EXPECT_EQ(fx.rules.begin_suppress_tokens, token_ids(hf.at("begin_suppress_tokens")));
  EXPECT_EQ(fx.rules.eos_token_id, hf.at("eos_token_id").as_int());
  EXPECT_EQ(fx.rules.max_length, hf.at("max_length").as_int());
  EXPECT_NO_THROW(kvb::check_whisper_rules(fx.rules, cfg));
}

TEST(WhisperContract, WeightFileIsAConvertedCheckpoint) {
  const Fixture fx;
  const kvb::WhisperConfig& cfg = fx.config;
  using Shape = std::vector<std::int64_t>;
  const auto shape = [&](const std::string& name) { return fx.weights.at(name).shape; };
  // Conv weights keep their three dimensions; position tables are stored, never recomputed.
  EXPECT_EQ(shape("model.encoder.conv1.weight"), (Shape{cfg.d_model, cfg.n_mels, 3}));
  EXPECT_EQ(shape("model.encoder.conv2.weight"), (Shape{cfg.d_model, cfg.d_model, 3}));
  EXPECT_EQ(shape("model.encoder.embed_positions.weight"),
            (Shape{cfg.max_source_positions, cfg.d_model}));
  EXPECT_EQ(shape("model.decoder.embed_positions.weight"),
            (Shape{cfg.max_target_positions, cfg.d_model}));
  EXPECT_EQ(shape("model.decoder.embed_tokens.weight"), (Shape{cfg.vocab_size, cfg.d_model}));
  // The output projection is the token embedding and is not stored.
  EXPECT_EQ(fx.weights.find("proj_out.weight"), nullptr);

  // Every attention has a key bias, all zeros: one per encoder layer and two per decoder layer.
  std::int64_t key_biases = 0;
  for (const auto& [name, tensor] : fx.weights.tensors()) {
    EXPECT_EQ(tensor.dtype, kvb::DType::F16) << name;
    if (!name.ends_with(".k_proj.bias")) continue;
    ++key_biases;
    const auto bits = tensor.as<std::uint16_t>();
    EXPECT_TRUE(std::all_of(bits.begin(), bits.end(), [](std::uint16_t b) { return b == 0; }))
        << name;
  }
  EXPECT_EQ(key_biases, cfg.encoder_layers + 2 * cfg.decoder_layers);
}

// Writes `text` to a file under a name unique to this test; removes it when destroyed.
class TempJson {
 public:
  explicit TempJson(const std::string& text) {
    const auto* test = testing::UnitTest::GetInstance()->current_test_info();
    path_ = std::filesystem::temp_directory_path() /
            (std::string("kvbound_") + test->test_suite_name() + "_" + test->name() + ".json");
    std::ofstream out(path_, std::ios::binary);
    out << text;
    out.close();
    EXPECT_TRUE(out.good()) << "cannot write " << path_;
  }
  ~TempJson() { std::filesystem::remove(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

TEST(WhisperContract, LoadersRejectWhatTheEngineDoesNotModel) {
  const std::string dims =
      R"("vocab_size": 8, "num_mel_bins": 4, "d_model": 8, "encoder_layers": 1,
         "encoder_attention_heads": 2, "encoder_ffn_dim": 16, "decoder_layers": 1,
         "decoder_attention_heads": 2, "decoder_ffn_dim": 16, "max_source_positions": 4,
         "max_target_positions": 4)";
  const auto load = [&](const std::string& head, const std::string& body) {
    const TempJson file("{" + head + ", " + body + "}");
    return kvb::load_whisper_config(file.path());
  };
  const std::string whisper = R"("model_type": "whisper")";
  EXPECT_EQ(load(whisper, dims).n_frames(), 8);
  EXPECT_EQ(load(whisper + R"(, "tie_word_embeddings": null)", dims).d_model, 8);
  EXPECT_THROW(load(R"("model_type": "llama")", dims), kvb::Error);
  EXPECT_THROW(load(whisper + R"(, "activation_function": "relu")", dims), kvb::Error);
  EXPECT_THROW(load(whisper + R"(, "scale_embedding": true)", dims), kvb::Error);
  EXPECT_THROW(load(whisper + R"(, "tie_word_embeddings": false)", dims), kvb::Error);
  EXPECT_THROW(load(whisper, R"("vocab_size": 8)"), kvb::Error);
  // Heads must divide the width.
  std::string odd = dims;
  odd.replace(odd.find(R"("d_model": 8)"), 12, R"("d_model": 9)");
  EXPECT_THROW(load(whisper, odd), kvb::Error);

  const auto rules = [&](const std::string& text) {
    const TempJson file(text);
    return kvb::load_whisper_rules(file.path());
  };
  const kvb::WhisperRules ok = rules(
      R"({"prompt": [1, 2], "suppress_tokens": [], "begin_suppress_tokens": [3],
          "eos_token_id": 0, "max_length": 4})");
  EXPECT_EQ(ok.prompt, (std::vector<std::int32_t>{1, 2}));
  EXPECT_TRUE(ok.suppress_tokens.empty());
  EXPECT_THROW(rules(R"({"prompt": [1], "suppress_tokens": [], "eos_token_id": 0})"), kvb::Error);
  EXPECT_THROW(rules(R"({"prompt": [-1], "suppress_tokens": [], "begin_suppress_tokens": [],
                         "eos_token_id": 0, "max_length": 4})"),
               kvb::Error);
}

class WhisperFusedOrNot : public testing::TestWithParam<bool> {};

TEST_P(WhisperFusedOrNot, ProgramsReproduceEveryGoldenSite) {
  const Fixture fx(GetParam());
  for (const std::string& prompt : fx.prompts()) {
    SCOPED_TRACE(prompt);
    CpuWhisper model(fx.config, fx.programs, fx.weights, 1);
    Capture capture;
    model.encode(fx.encoder_rows(0, 0), fx.mel(prompt), &capture);
    // Teacher-forced: every row is an output.
    const auto ids = fx.goldens.at(prompt + "/input_ids").as<std::int32_t>();
    std::vector<std::int32_t> out_rows(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) out_rows[i] = static_cast<std::int32_t>(i);
    model.decode(fx.decoder_rows(ids, 0), out_rows, &capture);
    fx.expect_goldens(capture, {prompt});
  }
}

TEST_P(WhisperFusedOrNot, ClipsOfOneStepDoNotMix) {
  const Fixture fx(GetParam());
  // Two clips in one encoder step, the first into cross block 1 and the second into block 0.
  // One decoder step then runs each clip's own token sequence, as the sequence that owns its
  // block, the rows of the second clip after the first's.
  const std::vector<std::string> prompts = fx.prompts();
  const std::vector<std::string> clips{prompts.at(1), prompts.at(0)};
  CpuWhisper model(fx.config, fx.programs, fx.weights, 2);
  std::vector<StepRow> rows;
  std::vector<float> mel;
  for (int clip = 0; clip < 2; ++clip) {
    const std::vector<StepRow> clip_rows = fx.encoder_rows(clip, 1 - clip);
    const std::vector<float> clip_mel = fx.mel(clips[clip]);
    rows.insert(rows.end(), clip_rows.begin(), clip_rows.end());
    mel.insert(mel.end(), clip_mel.begin(), clip_mel.end());
  }
  Capture capture;
  model.encode(rows, mel, &capture);

  rows.clear();
  std::vector<std::int32_t> out_rows;
  for (int clip = 0; clip < 2; ++clip) {
    const auto ids = fx.goldens.at(clips[clip] + "/input_ids").as<std::int32_t>();
    for (const StepRow& row : fx.decoder_rows(ids, 1 - clip)) {
      out_rows.push_back(static_cast<std::int32_t>(rows.size()));
      rows.push_back(row);
    }
  }
  model.decode(rows, out_rows, &capture);
  fx.expect_goldens(capture, clips);
}

TEST_P(WhisperFusedOrNot, GreedyDecodingTakesTheGoldenPath) {
  const Fixture fx(GetParam());
  const std::int32_t prompt_len = static_cast<std::int32_t>(fx.rules.prompt.size());
  bool stopped_at_eos = false, ran_to_max_length = false;
  for (const std::string& prompt : fx.prompts()) {
    SCOPED_TRACE(prompt);
    // The sequence holds block-table row 1; its clip is the only one of the encoder step.
    CpuWhisper model(fx.config, fx.programs, fx.weights, 2);
    model.encode(fx.encoder_rows(0, 1), fx.mel(prompt));
    // The prompt in one step with its last row as the output, then one token per step.
    std::vector<std::int32_t> ids = fx.rules.prompt;
    const std::vector<std::int32_t> last{prompt_len - 1};
    ids.push_back(model.decode(fx.decoder_rows(ids, 1), last).at(0));
    const std::vector<std::int32_t> only{0};
    while (ids.back() != fx.rules.eos_token_id &&
           static_cast<std::int64_t>(ids.size()) < fx.rules.max_length) {
      const std::int32_t position = static_cast<std::int32_t>(ids.size()) - 1;
      const std::vector<StepRow> row = fx.decoder_rows({&ids.back(), 1}, 1, position);
      ids.push_back(model.decode(row, only).at(0));
    }
    const auto golden = fx.goldens.at(prompt + "/greedy_ids").as<std::int32_t>();
    EXPECT_EQ(ids, std::vector<std::int32_t>(golden.begin(), golden.end()));
    stopped_at_eos = stopped_at_eos || ids.back() == fx.rules.eos_token_id;
    ran_to_max_length = ran_to_max_length || ids.back() != fx.rules.eos_token_id;
  }
  // The fixture's prompts cover both ways a transcription ends.
  EXPECT_TRUE(stopped_at_eos);
  EXPECT_TRUE(ran_to_max_length);
}

INSTANTIATE_TEST_SUITE_P(WhisperContract, WhisperFusedOrNot, testing::Bool());

}  // namespace
