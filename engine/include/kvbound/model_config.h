#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

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

// Architecture dimensions of a Whisper checkpoint, read from its Hugging Face config.json.
struct WhisperConfig {
  std::int64_t vocab_size = 0;
  std::int64_t d_model = 0;
  std::int64_t n_mels = 0;
  std::int64_t encoder_layers = 0;
  std::int64_t encoder_heads = 0;
  std::int64_t encoder_ffn_dim = 0;
  std::int64_t decoder_layers = 0;
  std::int64_t decoder_heads = 0;
  std::int64_t decoder_ffn_dim = 0;
  std::int64_t max_source_positions = 0;  // encoder positions of one audio window
  std::int64_t max_target_positions = 0;  // tokens a decoder sequence may hold
  double layer_norm_eps = 1e-5;

  // Mel frames of one audio window: the conv stem's stride of 2 halves them.
  std::int64_t n_frames() const noexcept { return 2 * max_source_positions; }
};

// Throws kvb::Error if the file is malformed or describes a variant the engine does not model
// (non-GELU activation, scaled embeddings, an output projection that is not the embedding).
WhisperConfig load_whisper_config(const std::filesystem::path& config_json);

// Greedy decoding rules of a Whisper checkpoint, read from its generation.json.
struct WhisperRules {
  // Tokens fed before the first generated one, starting with the decoder start token.
  std::vector<std::int32_t> prompt;
  std::vector<std::int32_t> suppress_tokens;        // never generated
  std::vector<std::int32_t> begin_suppress_tokens;  // not generated first after the prompt
  std::int32_t eos_token_id = -1;
  std::int64_t max_length = 0;  // largest length of the whole sequence, prompt included
};

// Throws kvb::Error if the file is malformed.
WhisperRules load_whisper_rules(const std::filesystem::path& generation_json);

// Throws kvb::Error unless the prompt is not empty, every token id is in the vocabulary and
// max_length is above the prompt length and within max_target_positions.
void check_whisper_rules(const WhisperRules& rules, const WhisperConfig& cfg);

}  // namespace kvb
