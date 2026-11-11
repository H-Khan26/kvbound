#include "kvbound/model_config.h"

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

}  // namespace

ModelConfig load_model_config(const std::filesystem::path& config_json) {
  const MappedFile file(config_json);
  try {
    return parse_model_config(json::parse(file.text()));
  } catch (const Error& e) {
    throw Error(config_json.string() + ": " + e.what());
  }
}

}  // namespace kvb
