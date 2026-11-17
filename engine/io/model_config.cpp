#include "kvbound/model_config.h"

#include <limits>
#include <string>

#include "kvbound/error.h"
#include "kvbound/json.h"
#include "kvbound/mapped_file.h"

namespace kvb {
namespace {

bool flag_or(const json::Value& root, std::string_view key, bool fallback) {
  const json::Value* value = root.find(key);
  return value != nullptr && !value->is_null() ? value->as_bool() : fallback;
}

ModelConfig parse_model_config(const json::Value& root) {
  if (root.at("model_type").as_string() != "llama") throw Error("model_type is not llama");
  if (const json::Value* act = root.find("hidden_act"); act && act->as_string() != "silu") {
    throw Error("unsupported hidden_act " + act->as_string());
  }
  if (flag_or(root, "attention_bias", false) || flag_or(root, "mlp_bias", false)) {
    throw Error("biased projections are not supported");
  }

  ModelConfig cfg;
  cfg.vocab_size = root.at("vocab_size").as_int();
  cfg.hidden_size = root.at("hidden_size").as_int();
  cfg.n_layers = root.at("num_hidden_layers").as_int();
  cfg.n_heads = root.at("num_attention_heads").as_int();
  const json::Value* kv_heads = root.find("num_key_value_heads");
  cfg.n_kv_heads = kv_heads ? kv_heads->as_int() : cfg.n_heads;
  cfg.intermediate_size = root.at("intermediate_size").as_int();
  cfg.max_position_embeddings = root.at("max_position_embeddings").as_int();
  cfg.rms_norm_eps = root.at("rms_norm_eps").as_number();
  const json::Value* theta = root.find("rope_theta");
  cfg.rope_theta = theta ? theta->as_number() : 10000.0;
  cfg.tie_word_embeddings = flag_or(root, "tie_word_embeddings", false);
  if (const json::Value* scaling = root.find("rope_scaling"); scaling && !scaling->is_null()) {
    if (scaling->at("rope_type").as_string() != "llama3") {
      throw Error("unsupported rope_type " + scaling->at("rope_type").as_string());
    }
    cfg.rope_scaling = Llama3RopeScaling{
        .factor = scaling->at("factor").as_number(),
        .low_freq_factor = scaling->at("low_freq_factor").as_number(),
        .high_freq_factor = scaling->at("high_freq_factor").as_number(),
        .original_max_position_embeddings =
            scaling->at("original_max_position_embeddings").as_int(),
    };
  }

  const bool positive = cfg.vocab_size > 0 && cfg.hidden_size > 0 && cfg.n_layers > 0 &&
                        cfg.n_heads > 0 && cfg.n_kv_heads > 0 && cfg.intermediate_size > 0;
  if (!positive || cfg.hidden_size % cfg.n_heads != 0 || cfg.n_heads % cfg.n_kv_heads != 0 ||
      cfg.head_dim() % 2 != 0) {
    throw Error("inconsistent dimensions");
  }
  if (const json::Value* head_dim = root.find("head_dim");
      head_dim && head_dim->as_int() != cfg.head_dim()) {
    throw Error("head_dim differs from hidden_size / num_attention_heads");
  }
  return cfg;
}

WhisperConfig parse_whisper_config(const json::Value& root) {
  if (root.at("model_type").as_string() != "whisper") throw Error("model_type is not whisper");
  if (const json::Value* act = root.find("activation_function");
      act && act->as_string() != "gelu") {
    throw Error("unsupported activation_function " + act->as_string());
  }
  if (flag_or(root, "scale_embedding", false)) throw Error("scaled embeddings are not supported");
  if (!flag_or(root, "tie_word_embeddings", true)) {
    throw Error("an output projection that is not the token embedding is not supported");
  }

  WhisperConfig cfg;
  cfg.vocab_size = root.at("vocab_size").as_int();
  cfg.d_model = root.at("d_model").as_int();
  cfg.n_mels = root.at("num_mel_bins").as_int();
  cfg.encoder_layers = root.at("encoder_layers").as_int();
  cfg.encoder_heads = root.at("encoder_attention_heads").as_int();
  cfg.encoder_ffn_dim = root.at("encoder_ffn_dim").as_int();
  cfg.decoder_layers = root.at("decoder_layers").as_int();
  cfg.decoder_heads = root.at("decoder_attention_heads").as_int();
  cfg.decoder_ffn_dim = root.at("decoder_ffn_dim").as_int();
  cfg.max_source_positions = root.at("max_source_positions").as_int();
  cfg.max_target_positions = root.at("max_target_positions").as_int();

  const bool positive = cfg.vocab_size > 0 && cfg.d_model > 0 && cfg.n_mels > 0 &&
                        cfg.encoder_layers > 0 && cfg.encoder_heads > 0 &&
                        cfg.encoder_ffn_dim > 0 && cfg.decoder_layers > 0 &&
                        cfg.decoder_heads > 0 && cfg.decoder_ffn_dim > 0 &&
                        cfg.max_source_positions > 0 && cfg.max_target_positions > 0;
  if (!positive || cfg.d_model % cfg.encoder_heads != 0 || cfg.d_model % cfg.decoder_heads != 0) {
    throw Error("inconsistent dimensions");
  }
  return cfg;
}

std::int32_t token_id(const json::Value& id) {
  const std::int64_t value = id.as_int();
  if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
    throw Error("token id " + std::to_string(value) + " is out of range");
  }
  return static_cast<std::int32_t>(value);
}

std::vector<std::int32_t> token_ids(const json::Value& list) {
  std::vector<std::int32_t> ids;
  for (const json::Value& id : list.as_array()) ids.push_back(token_id(id));
  return ids;
}

WhisperRules parse_whisper_rules(const json::Value& root) {
  WhisperRules rules;
  rules.prompt = token_ids(root.at("prompt"));
  rules.suppress_tokens = token_ids(root.at("suppress_tokens"));
  rules.begin_suppress_tokens = token_ids(root.at("begin_suppress_tokens"));
  rules.eos_token_id = token_id(root.at("eos_token_id"));
  rules.max_length = root.at("max_length").as_int();
  return rules;
}

// Parses the file with `parse_root`, prefixing any error with the file's path.
template <typename Parse>
auto load_json(const std::filesystem::path& path, Parse parse_root) {
  const MappedFile file(path);
  try {
    return parse_root(json::parse(file.text()));
  } catch (const Error& e) {
    throw Error(path.string() + ": " + e.what());
  }
}

}  // namespace

ModelConfig load_model_config(const std::filesystem::path& config_json) {
  return load_json(config_json, parse_model_config);
}

WhisperConfig load_whisper_config(const std::filesystem::path& config_json) {
  return load_json(config_json, parse_whisper_config);
}

WhisperRules load_whisper_rules(const std::filesystem::path& generation_json) {
  return load_json(generation_json, parse_whisper_rules);
}

void check_whisper_rules(const WhisperRules& rules, const WhisperConfig& cfg) {
  if (rules.prompt.empty()) throw Error("the decoding prompt is empty");
  const auto in_vocab = [&](std::int32_t token) { return token >= 0 && token < cfg.vocab_size; };
  for (const std::vector<std::int32_t>* ids :
       {&rules.prompt, &rules.suppress_tokens, &rules.begin_suppress_tokens}) {
    for (const std::int32_t token : *ids) {
      if (!in_vocab(token)) {
        throw Error("token id " + std::to_string(token) + " is outside the vocabulary");
      }
    }
  }
  if (!in_vocab(rules.eos_token_id)) throw Error("eos_token_id is outside the vocabulary");
  const std::int64_t prompt_len = static_cast<std::int64_t>(rules.prompt.size());
  if (rules.max_length <= prompt_len || rules.max_length > cfg.max_target_positions) {
    throw Error("max_length " + std::to_string(rules.max_length) +
                " is not above the prompt length and within max_target_positions " +
                std::to_string(cfg.max_target_positions));
  }
}

}  // namespace kvb
