#include "kvbound/program.h"

#include <array>

namespace kvb {

std::string_view op_name(OpKind kind) noexcept {
  constexpr std::array<std::string_view, kOpKindCount> names{
      "embed",    "rms_norm", "linear",      "rope",   "kv_write",     "attention",
      "silu_mul", "add",      "gather_rows", "argmax", "add_rms_norm", "rope_kv_write"};
  return names[static_cast<std::size_t>(kind)];
}

namespace {

// Appends values, weights and ops, handing back their ids.
struct Builder {
  Program program;

  int value(std::string name, RowKind rows, std::int64_t cols, DType dtype = DType::F16) {
    program.values.push_back({std::move(name), rows, cols, dtype});
    return static_cast<int>(program.values.size()) - 1;
  }
  int weight(WeightParts parts) {
    program.weights.push_back(std::move(parts));
    return static_cast<int>(program.weights.size()) - 1;
  }
  void op(Op op) { program.ops.push_back(std::move(op)); }
};

std::string layer_site(int i) { return "layers." + std::to_string(i); }
std::string layer_weight(int i) { return "model.layers." + std::to_string(i); }

}  // namespace

Program build_llama_program(const ModelConfig& cfg, bool fuse) {
  Builder b;
  const std::int64_t hidden = cfg.hidden_size;
  const std::int64_t kv_dim = cfg.n_kv_heads * cfg.head_dim();
  const int n_layers = static_cast<int>(cfg.n_layers);
  const int embed_weight = b.weight({"model.embed_tokens.weight"});

  // The residual stream: written by the embedding, then updated in place by every layer.
  const int h = b.value("h", RowKind::Token, hidden);
  b.op({.kind = OpKind::Embed, .out = h, .weight = embed_weight, .site = "embed"});

  // The input norm of a layer. Fused layers compute it at the end of the layer before, so only
  // the first one stands alone.
  int attn_norm = -1;
  const auto input_norm = [&](int i) {
    attn_norm = b.value(layer_site(i) + ".attn_norm", RowKind::Token, hidden);
    b.op({.kind = OpKind::RmsNorm,
          .layer = i,
          .in0 = h,
          .out = attn_norm,
          .weight = b.weight({layer_weight(i) + ".input_layernorm.weight"}),
          .site = layer_site(i) + ".attn_norm"});
  };
  if (fuse && n_layers > 0) input_norm(0);

  for (int i = 0; i < n_layers; ++i) {
    const std::string site = layer_site(i);
    const std::string w = layer_weight(i);
    const auto name = [&](std::string_view suffix) { return site + "." + std::string(suffix); };
    const auto linear = [&](int in, int out, WeightParts weight, std::string capture = {}) {
      b.op({.kind = OpKind::Linear,
            .layer = i,
            .in0 = in,
            .out = out,
            .weight = b.weight(std::move(weight)),
            .site = std::move(capture)});
    };
    if (!fuse) input_norm(i);

    const int q = b.value(name("q"), RowKind::Token, hidden);
    if (fuse) {
      const int qkv = b.value(name("qkv"), RowKind::Token, hidden + 2 * kv_dim);
      linear(attn_norm, qkv,
             {w + ".self_attn.q_proj.weight", w + ".self_attn.k_proj.weight",
              w + ".self_attn.v_proj.weight"});
      b.op({.kind = OpKind::RopeKvWrite, .layer = i, .in0 = qkv, .out = q});
    } else {
      const int k = b.value(name("k"), RowKind::Token, kv_dim);
      const int v = b.value(name("v"), RowKind::Token, kv_dim);
      linear(attn_norm, q, {w + ".self_attn.q_proj.weight"});
      linear(attn_norm, k, {w + ".self_attn.k_proj.weight"});
      linear(attn_norm, v, {w + ".self_attn.v_proj.weight"});
      b.op({.kind = OpKind::Rope, .layer = i, .in0 = q, .out = q});
      b.op({.kind = OpKind::Rope, .layer = i, .in0 = k, .out = k});
      // The step's own keys enter the cache first, so attention reads every key from one place.
      b.op({.kind = OpKind::KvWrite, .layer = i, .in0 = k, .in1 = v});
    }

    const int mixed = b.value(name("attn_mixed"), RowKind::Token, hidden);
    b.op({.kind = OpKind::Attention, .layer = i, .in0 = q, .out = mixed});
    const int attn_out = b.value(name("attn_out"), RowKind::Token, hidden);
    linear(mixed, attn_out, {w + ".self_attn.o_proj.weight"}, name("attn_out"));

    const int mlp_norm = b.value(name("mlp_norm"), RowKind::Token, hidden);
    const int mlp_norm_weight = b.weight({w + ".post_attention_layernorm.weight"});
    if (fuse) {
      b.op({.kind = OpKind::AddRmsNorm,
            .layer = i,
            .in0 = h,
            .in1 = attn_out,
            .out = mlp_norm,
            .weight = mlp_norm_weight,
            .site = name("mlp_norm"),
            .in0_site = name("resid_mid")});
    } else {
      b.op({.kind = OpKind::Add,
            .layer = i,
            .in0 = h,
            .in1 = attn_out,
            .out = h,
            .site = name("resid_mid")});
      b.op({.kind = OpKind::RmsNorm,
            .layer = i,
            .in0 = h,
            .out = mlp_norm,
            .weight = mlp_norm_weight,
            .site = name("mlp_norm")});
    }

    const int gate = b.value(name("gate"), RowKind::Token, cfg.intermediate_size);
    if (fuse) {
      const int gate_up = b.value(name("gate_up"), RowKind::Token, 2 * cfg.intermediate_size);
      linear(mlp_norm, gate_up, {w + ".mlp.gate_proj.weight", w + ".mlp.up_proj.weight"});
      b.op({.kind = OpKind::SiluMul, .layer = i, .in0 = gate_up, .out = gate});
    } else {
      const int up = b.value(name("up"), RowKind::Token, cfg.intermediate_size);
      linear(mlp_norm, gate, {w + ".mlp.gate_proj.weight"});
      linear(mlp_norm, up, {w + ".mlp.up_proj.weight"});
      b.op({.kind = OpKind::SiluMul, .layer = i, .in0 = gate, .in1 = up, .out = gate});
    }
    const int mlp_out = b.value(name("mlp_out"), RowKind::Token, hidden);
    linear(gate, mlp_out, {w + ".mlp.down_proj.weight"}, name("mlp_out"));

    if (fuse && i + 1 < n_layers) {
      const std::string next = layer_site(i + 1);
      attn_norm = b.value(next + ".attn_norm", RowKind::Token, hidden);
      b.op({.kind = OpKind::AddRmsNorm,
            .layer = i,
            .in0 = h,
            .in1 = mlp_out,
            .out = attn_norm,
            .weight = b.weight({layer_weight(i + 1) + ".input_layernorm.weight"}),
            .site = next + ".attn_norm",
            .in0_site = name("resid_post")});
    } else {
      b.op({.kind = OpKind::Add,
            .layer = i,
            .in0 = h,
            .in1 = mlp_out,
            .out = h,
            .site = name("resid_post")});
    }
  }

  // Only rows whose next token is wanted go through the final norm and the vocabulary projection.
  const int last = b.value("h_out", RowKind::Output, hidden);
  b.op({.kind = OpKind::GatherRows, .in0 = h, .out = last});
  const int final_norm = b.value("final_norm", RowKind::Output, hidden);
  b.op({.kind = OpKind::RmsNorm,
        .in0 = last,
        .out = final_norm,
        .weight = b.weight({"model.norm.weight"}),
        .site = "final_norm"});
  // FP32 logits: FP16 spacing near typical logit magnitudes is coarse enough to create ties.
  b.program.logits = b.value("logits", RowKind::Output, cfg.vocab_size, DType::F32);
  b.op({.kind = OpKind::Linear,
        .in0 = final_norm,
        .out = b.program.logits,
        .weight = cfg.tie_word_embeddings ? embed_weight : b.weight({"lm_head.weight"}),
        .site = "logits"});
  b.program.next_ids = b.value("next_ids", RowKind::Output, 1, DType::I32);
  b.op({.kind = OpKind::Argmax, .in0 = b.program.logits, .out = b.program.next_ids});
  return std::move(b.program);
}

}  // namespace kvb
