#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

namespace kvb {

// Llama 3 RoPE frequency scaling: rotations with a wavelength above
// original_max_position_embeddings / low_freq_factor are slowed by `factor`, those below
// original_max_position_embeddings / high_freq_factor are kept, and the band between blends.
struct Llama3RopeScaling {
  double factor = 1.0;
  double low_freq_factor = 1.0;
  double high_freq_factor = 1.0;
  std::int64_t original_max_position_embeddings = 0;
};

// Architecture dimensions of a Llama-family checkpoint, read from its Hugging Face config.json.
struct ModelConfig {
  std::int64_t vocab_size = 0;
  std::int64_t hidden_size = 0;
  std::int64_t n_layers = 0;
  std::int64_t n_heads = 0;
  std::int64_t n_kv_heads = 0;
  std::int64_t intermediate_size = 0;
  std::int64_t max_position_embeddings = 0;
  double rms_norm_eps = 0.0;
  double rope_theta = 0.0;
  bool tie_word_embeddings = false;
  std::optional<Llama3RopeScaling> rope_scaling;

  std::int64_t head_dim() const noexcept { return hidden_size / n_heads; }
  // Query heads served by each KV head (grouped-query attention).
  std::int64_t n_rep() const noexcept { return n_heads / n_kv_heads; }
};

// Throws kvb::Error if the file is malformed or describes a variant the engine does not model
// (non-SiLU activation, RoPE scaling other than llama3, biased projections).
ModelConfig load_model_config(const std::filesystem::path& config_json);

}  // namespace kvb
