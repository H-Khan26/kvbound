// Proves the Python-to-C++ contract end to end on the committed fixture: the weight file format,
// the tensor naming, the golden layout and the manifest thresholds. A CPU RMSNorm stands in for
// the engine: it reads weights and golden inputs through the real loader and must reproduce the
// golden outputs within the manifest's FP32 thresholds.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "kvbound/dtype.h"
#include "kvbound/json.h"
#include "kvbound/mapped_file.h"
#include "kvbound/model_config.h"
#include "kvbound/safetensors.h"
#include "support/tolerance.h"

namespace {

const std::filesystem::path kFixture = KVB_FIXTURE_DIR;

// y = x * rsqrt(mean(x^2) + eps) * weight, applied to each row of x [rows, dim].
std::vector<float> rms_norm(std::span<const float> x, std::span<const float> weight, double eps) {
  const std::size_t dim = weight.size();
  std::vector<float> y(x.size());
  for (std::size_t row = 0; row < x.size() / dim; ++row) {
    const std::span<const float> in = x.subspan(row * dim, dim);
    float sum_sq = 0.0f;
    for (const float v : in) sum_sq += v * v;
    const float scale =
        1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + static_cast<float>(eps));
    for (std::size_t i = 0; i < dim; ++i) y[row * dim + i] = in[i] * scale * weight[i];
  }
  return y;
}

std::vector<float> f16_to_f32(const kvb::TensorView& tensor) {
  std::vector<float> out;
  for (const std::uint16_t bits : tensor.as<std::uint16_t>()) {
    out.push_back(kvb::half_bits_to_float(bits));
  }
  return out;
}

struct Fixture {
  kvb::ModelConfig config = kvb::load_model_config(kFixture / "config.json");
  kvb::SafetensorsFile weights{kFixture / "model.safetensors"};
  kvb::SafetensorsFile goldens{kFixture / "goldens.safetensors"};
  kvb::MappedFile manifest_file{kFixture / "manifest.json"};
  kvb::json::Value manifest = kvb::json::parse(manifest_file.text());
};

TEST(FixtureContract, WeightFileMatchesConfig) {
  const Fixture fx;
  const kvb::ModelConfig& cfg = fx.config;
  EXPECT_EQ(fx.weights.metadata().at("format"), "pt");
  ASSERT_TRUE(cfg.rope_scaling.has_value());
  EXPECT_EQ(cfg.rope_scaling->factor, 8.0);

  const kvb::TensorView& embed = fx.weights.at("model.embed_tokens.weight");
  EXPECT_EQ(embed.dtype, kvb::DType::F16);
  EXPECT_EQ(embed.shape, (std::vector<std::int64_t>{cfg.vocab_size, cfg.hidden_size}));

  const std::int64_t kv_dim = cfg.n_kv_heads * cfg.head_dim();
  for (std::int64_t i = 0; i < cfg.n_layers; ++i) {
    const std::string layer = "model.layers." + std::to_string(i);
    EXPECT_EQ(fx.weights.at(layer + ".self_attn.k_proj.weight").shape,
              (std::vector<std::int64_t>{kv_dim, cfg.hidden_size}));
    EXPECT_EQ(fx.weights.at(layer + ".mlp.gate_proj.weight").shape,
              (std::vector<std::int64_t>{cfg.intermediate_size, cfg.hidden_size}));
  }
  // Tied embeddings: the output projection is not stored separately.
  EXPECT_EQ(fx.weights.find("lm_head.weight") == nullptr, cfg.tie_word_embeddings);
}

TEST(FixtureContract, RmsNormReproducesGoldens) {
  const Fixture fx;
  const kvb::ModelConfig& cfg = fx.config;
  const kvb::json::Value& fp32 = fx.manifest.at("thresholds").at("fp32");
  const double rel_rms_max = fp32.at("rel_rms_max").as_number();
  const double cosine_min = fp32.at("cosine_min").as_number();
  ASSERT_EQ(fx.manifest.at("golden_dtype").as_string(), "F32");

  // Checks norm(weight_name, golden input site) against a golden output site for one prompt.
  const auto check = [&](const std::string& prompt, const std::string& weight_name,
                         const std::string& input_site, const std::string& output_site) {
    SCOPED_TRACE(prompt + "/" + output_site);
    const std::vector<float> weight = f16_to_f32(fx.weights.at(weight_name));
    const auto input = fx.goldens.at(prompt + "/" + input_site).as<float>();
    const auto expected = fx.goldens.at(prompt + "/" + output_site).as<float>();
    ASSERT_EQ(weight.size(), static_cast<std::size_t>(cfg.hidden_size));
    ASSERT_EQ(input.size(), expected.size());

    const std::vector<float> actual = rms_norm(input, weight, cfg.rms_norm_eps);
    EXPECT_LE(kvb::test::rel_rms(actual, expected), rel_rms_max);
    EXPECT_GE(kvb::test::cosine(actual, expected), cosine_min);
  };

  const kvb::json::Array& prompts = fx.manifest.at("prompts").as_array();
  ASSERT_FALSE(prompts.empty());
  for (const kvb::json::Value& entry : prompts) {
    const std::string& prompt = entry.at("id").as_string();
    const kvb::TensorView& ids = fx.goldens.at(prompt + "/input_ids");
    EXPECT_EQ(ids.dtype, kvb::DType::I32);
    EXPECT_EQ(static_cast<std::int64_t>(ids.numel()), entry.at("n_tokens").as_int());

    std::string residual = "embed";
    for (std::int64_t i = 0; i < cfg.n_layers; ++i) {
      const std::string site = "layers." + std::to_string(i);
      const std::string weights = "model.layers." + std::to_string(i);
      check(prompt, weights + ".input_layernorm.weight", residual, site + ".attn_norm");
      check(prompt, weights + ".post_attention_layernorm.weight", site + ".resid_mid",
            site + ".mlp_norm");
      residual = site + ".resid_post";
    }
    check(prompt, "model.norm.weight", residual, "final_norm");
  }
}

}  // namespace
