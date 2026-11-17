#include "kvbound/program.h"

#include <initializer_list>
#include <string>
#include <utility>

namespace kvb {
namespace {

// A weight and its bias, as indices into the weight table.
struct Affine {
  int weight = -1;
  int bias = -1;
};

// The weight table both programs share. Entries are added in forward order, encoder first.
struct Weights {
  std::vector<WeightParts> parts;
  std::vector<std::vector<std::int64_t>> part_rows;

  int add(WeightParts names, std::vector<std::int64_t> rows = {}) {
    parts.push_back(std::move(names));
    part_rows.push_back(std::move(rows));
    return static_cast<int>(parts.size()) - 1;
  }
  // The weight and bias tensors of one module.
  Affine affine(const std::string& module) {
    return {add({module + ".weight"}), add({module + ".bias"})};
  }
  // Modules that read the same input, stacked along their outputs, `rows` outputs each.
  Affine stacked(std::initializer_list<std::string> modules, std::int64_t rows) {
    WeightParts weights, biases;
    for (const std::string& module : modules) {
      weights.push_back(module + ".weight");
      biases.push_back(module + ".bias");
    }
    const std::vector<std::int64_t> each(modules.size(), rows);
    return {add(std::move(weights), each), add(std::move(biases), each)};
  }
};

// Appends values and ops to one program. `h` is the residual stream, updated in place by every
// layer. With `fuse`, an add into it is held back until the norm that follows, and the two
// become one op.
struct Builder {
  Program program;
  Weights& weights;
  bool fuse = false;
  int h = -1;
  int held = -1;  // value of the add held back, -1 if none
  int held_layer = -1;
  std::string held_site;

  int value(std::string name, RowKind rows, std::int64_t cols, DType dtype = DType::F16) {
    program.values.push_back({std::move(name), rows, cols, dtype});
    return static_cast<int>(program.values.size()) - 1;
  }
  void op(Op op) { program.ops.push_back(std::move(op)); }

  void linear(int layer, int in, int out, Affine w, std::string site = {}) {
    op({.kind = OpKind::Linear,
        .layer = layer,
        .in0 = in,
        .out = out,
        .weight = w.weight,
        .bias = w.bias,
        .site = std::move(site)});
  }
  // h += value. `site` names h after the add.
  void add(int layer, int addend, std::string site) {
    flush();
    held = addend;
    held_layer = layer;
    held_site = std::move(site);
    if (!fuse) flush();
  }
  // Emits the add held back, for a reader of h that is not a norm.
  void flush() {
    if (held < 0) return;
    op({.kind = OpKind::Add,
        .layer = held_layer,
        .in0 = h,
        .in1 = held,
        .out = h,
        .site = std::move(held_site)});
    held = -1;
  }
  // A new token-row value `name`, also its capture site: the layer norm of h.
  int norm(int layer, const std::string& name, Affine w) {
    const int out = value(name, RowKind::Token, program.values[h].cols);
    Op layer_norm{.kind = OpKind::LayerNorm,
                  .layer = layer,
                  .in0 = h,
                  .out = out,
                  .weight = w.weight,
                  .bias = w.bias,
                  .site = name};
    if (held >= 0) {
      layer_norm.kind = OpKind::AddLayerNorm;
      layer_norm.layer = held_layer;
      layer_norm.in1 = std::exchange(held, -1);
      layer_norm.in0_site = std::move(held_site);
    }
    op(std::move(layer_norm));
    return out;
  }
  // h += fc2(gelu(fc1(in))), the feed-forward half of the layer whose sites start with `site`
  // and whose weights start with `module`.
  void mlp(int layer, const std::string& site, const std::string& module, int in,
           std::int64_t ffn_dim) {
    const int fc1 = value(site + ".fc1", RowKind::Token, ffn_dim);
    linear(layer, in, fc1, weights.affine(module + ".fc1"));
    op({.kind = OpKind::Gelu, .layer = layer, .in0 = fc1, .out = fc1});
    const int mlp_out = value(site + ".mlp_out", RowKind::Token, program.values[h].cols);
    linear(layer, fc1, mlp_out, weights.affine(module + ".fc2"), site + ".mlp_out");
    add(layer, mlp_out, site + ".resid_post");
  }
};

Program build_encoder(const WhisperConfig& cfg, Weights& weights, bool fuse) {
  Builder b{.weights = weights, .fuse = fuse};
  const std::int64_t d = cfg.d_model;
  const std::string model = "model.encoder";

  // The conv stem. Each convolution is an unfold of three neighbouring frames, then a product.
  const int mel = b.value("encoder.mel", RowKind::Frame, cfg.n_mels);
  b.program.input = mel;
  const int conv1_in = b.value("encoder.conv1_in", RowKind::Frame, 3 * cfg.n_mels);
  const int conv1 = b.value("encoder.conv1", RowKind::Frame, d);
  b.op({.kind = OpKind::Unfold, .in0 = mel, .out = conv1_in});
  b.linear(-1, conv1_in, conv1, weights.affine(model + ".conv1"));
  b.op({.kind = OpKind::Gelu, .in0 = conv1, .out = conv1, .site = "encoder.conv1"});
  // The second convolution has stride 2: two frames in, one position out.
  const int conv2_in = b.value("encoder.conv2_in", RowKind::Token, 3 * d);
  b.h = b.value("encoder.h", RowKind::Token, d);
  b.op({.kind = OpKind::Unfold, .in0 = conv1, .out = conv2_in});
  b.linear(-1, conv2_in, b.h, weights.affine(model + ".conv2"));
  b.op({.kind = OpKind::Gelu, .in0 = b.h, .out = b.h, .site = "encoder.conv2"});
  b.op({.kind = OpKind::AddPositions,
        .in0 = b.h,
        .out = b.h,
        .weight = weights.add({model + ".embed_positions.weight"}),
        .site = "encoder.embed"});

  for (int i = 0; i < static_cast<int>(cfg.encoder_layers); ++i) {
    const std::string site = "encoder.layers." + std::to_string(i);
    const std::string module = model + ".layers." + std::to_string(i);
    const std::string attn = module + ".self_attn";

    const int attn_norm =
        b.norm(i, site + ".attn_norm", weights.affine(module + ".self_attn_layer_norm"));
    // One product computes q, k and v, and attention reads the keys and values from it: every
    // row of a clip is in the step, so nothing is cached.
    const int qkv = b.value(site + ".qkv", RowKind::Token, 3 * d);
    b.linear(i, attn_norm, qkv,
             weights.stacked({attn + ".q_proj", attn + ".k_proj", attn + ".v_proj"}, d));
    const int mixed = b.value(site + ".attn_mixed", RowKind::Token, d);
    b.op({.kind = OpKind::DenseAttention, .layer = i, .in0 = qkv, .out = mixed});
    const int attn_out = b.value(site + ".attn_out", RowKind::Token, d);
    b.linear(i, mixed, attn_out, weights.affine(attn + ".out_proj"), site + ".attn_out");
    b.add(i, attn_out, site + ".resid_mid");

    const int mlp_norm =
        b.norm(i, site + ".mlp_norm", weights.affine(module + ".final_layer_norm"));
    b.mlp(i, site, module, mlp_norm, cfg.encoder_ffn_dim);
  }
  const int final_norm = b.norm(-1, "encoder.final_norm", weights.affine(model + ".layer_norm"));

  // The cross-attention keys and values of every decoder layer are computed once per clip.
  for (int j = 0; j < static_cast<int>(cfg.decoder_layers); ++j) {
    const std::string name = "encoder.cross_kv." + std::to_string(j);
    const std::string attn = "model.decoder.layers." + std::to_string(j) + ".encoder_attn";
    const int kv = b.value(name, RowKind::Token, 2 * d);
    b.linear(j, final_norm, kv, weights.stacked({attn + ".k_proj", attn + ".v_proj"}, d), name);
    b.op({.kind = OpKind::KvWrite, .layer = j, .in0 = kv, .cache = KvCache::Cross});
  }
  return std::move(b.program);
}

Program build_decoder(const WhisperConfig& cfg, const WhisperRules& rules, Weights& weights,
                      bool fuse) {
  Builder b{.weights = weights, .fuse = fuse};
  const std::int64_t d = cfg.d_model;
  const std::string model = "model.decoder";
  const int embed_weight = weights.add({model + ".embed_tokens.weight"});

  b.h = b.value("decoder.h", RowKind::Token, d);
  b.op({.kind = OpKind::Embed, .out = b.h, .weight = embed_weight});
  b.op({.kind = OpKind::AddPositions,
        .in0 = b.h,
        .out = b.h,
        .weight = weights.add({model + ".embed_positions.weight"}),
        .site = "decoder.embed"});

  for (int i = 0; i < static_cast<int>(cfg.decoder_layers); ++i) {
    const std::string site = "decoder.layers." + std::to_string(i);
    const std::string module = model + ".layers." + std::to_string(i);
    const std::string attn = module + ".self_attn";
    const std::string cross = module + ".encoder_attn";
    const auto token_value = [&](std::string_view suffix) {
      return b.value(site + "." + std::string(suffix), RowKind::Token, d);
    };

    const int attn_norm =
        b.norm(i, site + ".attn_norm", weights.affine(module + ".self_attn_layer_norm"));
    const int q = token_value("q"), k = token_value("k"), v = token_value("v");
    b.linear(i, attn_norm, q, weights.affine(attn + ".q_proj"));
    b.linear(i, attn_norm, k, weights.affine(attn + ".k_proj"));
    b.linear(i, attn_norm, v, weights.affine(attn + ".v_proj"));
    // The step's own keys enter the cache first, so attention reads every key from one place.
    b.op({.kind = OpKind::KvWrite, .layer = i, .in0 = k, .in1 = v});
    const int mixed = token_value("attn_mixed");
    b.op({.kind = OpKind::Attention, .layer = i, .in0 = q, .out = mixed});
    const int attn_out = token_value("attn_out");
    b.linear(i, mixed, attn_out, weights.affine(attn + ".out_proj"), site + ".attn_out");
    b.add(i, attn_out, site + ".resid_mid");

    // Cross attention projects only its queries: the encoder program wrote the keys and values.
    const int cross_norm =
        b.norm(i, site + ".cross_norm", weights.affine(module + ".encoder_attn_layer_norm"));
    const int cross_q = token_value("cross_q");
    b.linear(i, cross_norm, cross_q, weights.affine(cross + ".q_proj"));
    const int cross_mixed = token_value("cross_mixed");
    b.op({.kind = OpKind::Attention,
          .layer = i,
          .in0 = cross_q,
          .out = cross_mixed,
          .cache = KvCache::Cross});
    const int cross_out = token_value("cross_out");
    b.linear(i, cross_mixed, cross_out, weights.affine(cross + ".out_proj"), site + ".cross_out");
    b.add(i, cross_out, site + ".resid_cross");

    const int mlp_norm =
        b.norm(i, site + ".mlp_norm", weights.affine(module + ".final_layer_norm"));
    b.mlp(i, site, module, mlp_norm, cfg.decoder_ffn_dim);
  }

  // Only rows whose next token is wanted go through the final norm and the vocabulary projection.
  b.flush();
  const int last = b.value("decoder.h_out", RowKind::Output, d);
  b.op({.kind = OpKind::GatherRows, .in0 = b.h, .out = last});
  const Affine norm = weights.affine(model + ".layer_norm");
  const int final_norm = b.value("decoder.final_norm", RowKind::Output, d);
  b.op({.kind = OpKind::LayerNorm,
        .in0 = last,
        .out = final_norm,
        .weight = norm.weight,
        .bias = norm.bias,
        .site = "decoder.final_norm"});
  // The output projection is the token embedding, without a bias. FP32 logits: FP16 spacing
  // near typical logit magnitudes is coarse enough to create ties.
  b.program.logits = b.value("logits", RowKind::Output, cfg.vocab_size, DType::F32);
  b.op({.kind = OpKind::Linear,
        .in0 = final_norm,
        .out = b.program.logits,
        .weight = embed_weight,
        .site = "logits"});
  b.op({.kind = OpKind::SuppressTokens, .in0 = b.program.logits, .out = b.program.logits});
  b.program.suppress = {.always = rules.suppress_tokens,
                        .begin = rules.begin_suppress_tokens,
                        .begin_position = static_cast<std::int32_t>(rules.prompt.size()) - 1};
  b.program.next_ids = b.value("next_ids", RowKind::Output, 1, DType::I32);
  b.op({.kind = OpKind::Argmax, .in0 = b.program.logits, .out = b.program.next_ids});
  return std::move(b.program);
}

}  // namespace

WhisperPrograms build_whisper_programs(const WhisperConfig& cfg, const WhisperRules& rules,
                                       bool fuse) {
  check_whisper_rules(rules, cfg);
  Weights weights;
  WhisperPrograms programs;
  programs.encoder = build_encoder(cfg, weights, fuse);
  programs.decoder = build_decoder(cfg, rules, weights, fuse);
  for (Program* program : {&programs.encoder, &programs.decoder}) {
    program->weights = weights.parts;
    program->part_rows = weights.part_rows;
  }
  return programs;
}

}  // namespace kvb
