"""From-scratch Whisper reference model: conv stem, pre-LN encoder, decoder with cross attention.

This is the ground truth the engine's speech path is tested against. It uses no Hugging Face
modeling code and loads weights straight from a safetensors file. Every intermediate the engine
must reproduce can be captured by name (see site_names).

The checkpoint is a converted one (kvbound.models.convert_whisper): Hugging Face tensor names,
an explicit zero bias for every k_proj, and no proj_out tensor, because the output projection
is the token embedding.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from kvbound.ref.checkpoint import WeightPrecision, load_weights
from kvbound.ref.config import WhisperConfig

# Capture sites of one layer, in forward order.
ENCODER_LAYER_SITES = ("attn_norm", "attn_out", "resid_mid", "mlp_norm", "mlp_out", "resid_post")
DECODER_LAYER_SITES = (
    "attn_norm",
    "attn_out",
    "resid_mid",
    "cross_norm",
    "cross_out",
    "resid_cross",
    "mlp_norm",
    "mlp_out",
    "resid_post",
)
# Site patterns in forward order: "layers.{i}.*" repeats per layer and "cross_kv.{j}" per decoder
# layer. Encoder sites have one row per frame, decoder sites one row per token.
#   encoder.conv1     GELU(conv1(features)), [2 * max_source_positions, d_model]
#   encoder.conv2     GELU(conv2(conv1 site)), [max_source_positions, d_model]
#   encoder.embed     conv2 site + position table
#   encoder.cross_kv  keys then values of decoder layer j's cross attention, [.., 2 * d_model]
#   decoder.embed     token embedding + position table
SITES = (
    "encoder.conv1",
    "encoder.conv2",
    "encoder.embed",
    *(f"encoder.layers.{{i}}.{s}" for s in ENCODER_LAYER_SITES),
    "encoder.final_norm",
    "encoder.cross_kv.{j}",
    "decoder.embed",
    *(f"decoder.layers.{{i}}.{s}" for s in DECODER_LAYER_SITES),
    "decoder.final_norm",
    "logits",
)

Capture = dict[str, Tensor]
# Keys and values of each decoder layer's cross attention, each [B, max_source_positions, d_model].
CrossKv = list[tuple[Tensor, Tensor]]


def site_names(cfg: WhisperConfig) -> list[str]:
    """Every capture site of a model, in forward order."""
    encoder, decoder = (
        [f"{side}.layers.{i}.{s}" for i in range(n) for s in sites]
        for side, n, sites in (
            ("encoder", cfg.encoder_layers, ENCODER_LAYER_SITES),
            ("decoder", cfg.decoder_layers, DECODER_LAYER_SITES),
        )
    )
    return [
        "encoder.conv1",
        "encoder.conv2",
        "encoder.embed",
        *encoder,
        "encoder.final_norm",
        *(f"encoder.cross_kv.{j}" for j in range(cfg.decoder_layers)),
        "decoder.embed",
        *decoder,
        "decoder.final_norm",
        "logits",
    ]


def sinusoids(length: int, channels: int) -> Tensor:
    """Whisper's fixed encoder position table [length, channels]: sines, then cosines.

    Channel j of each half has timescale 10000^(j / (channels/2 - 1)).
    """
    increment = math.log(10000.0) / (channels // 2 - 1)
    inv_timescales = torch.exp(-increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat((scaled_time.sin(), scaled_time.cos()), dim=1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Mean and variance run in FP32 whatever the model precision, as the engine's kernel does.
        y = F.layer_norm(x.float(), x.shape[-1:], self.weight.float(), self.bias.float(), self.eps)
        return y.to(x.dtype)


class Attention(nn.Module):
    """Multi-head attention whose keys and values come from kv(source).

    Self attention passes its own input as the source; cross attention passes the encoder output.
    A Whisper checkpoint has no key bias; the converted file stores a zero one, so every
    projection reads the same way.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def kv(self, source: Tensor) -> tuple[Tensor, Tensor]:
        return self.k_proj(source), self.v_proj(source)

    def split(self, x: Tensor) -> Tensor:
        """[B, T, d_model] to [B, heads, T, head_dim]."""
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.split(self.q_proj(x)), self.split(k), self.split(v)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if causal:
            future = torch.ones(t, k.shape[2], dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        # Softmax runs in FP32 whatever the model precision, as the engine's kernel does.
        probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
        return self.out_proj((probs @ v).transpose(1, 2).reshape(b, t, d))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        self.attn_norm = LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        self.attn = Attention(cfg.d_model, cfg.encoder_heads)
        self.mlp_norm = LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)

    def forward(self, x: Tensor, sites: Capture) -> Tensor:
        h = sites["attn_norm"] = self.attn_norm(x)
        sites["attn_out"] = self.attn(h, *self.attn.kv(h), causal=False)
        sites["resid_mid"] = x + sites["attn_out"]
        sites["mlp_norm"] = self.mlp_norm(sites["resid_mid"])
        sites["mlp_out"] = self.fc2(F.gelu(self.fc1(sites["mlp_norm"])))
        sites["resid_post"] = sites["resid_mid"] + sites["mlp_out"]
        return sites["resid_post"]


class DecoderBlock(nn.Module):
    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        self.attn_norm = LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        self.attn = Attention(cfg.d_model, cfg.decoder_heads)
        self.cross_norm = LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        self.cross_attn = Attention(cfg.d_model, cfg.decoder_heads)
        self.mlp_norm = LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        self.fc1 = nn.Linear(cfg.d_model, cfg.decoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.decoder_ffn_dim, cfg.d_model)

    def forward(self, x: Tensor, cross_k: Tensor, cross_v: Tensor, sites: Capture) -> Tensor:
        h = sites["attn_norm"] = self.attn_norm(x)
        sites["attn_out"] = self.attn(h, *self.attn.kv(h), causal=True)
        sites["resid_mid"] = x + sites["attn_out"]
        sites["cross_norm"] = self.cross_norm(sites["resid_mid"])
        sites["cross_out"] = self.cross_attn(sites["cross_norm"], cross_k, cross_v, causal=False)
        sites["resid_cross"] = sites["resid_mid"] + sites["cross_out"]
        sites["mlp_norm"] = self.mlp_norm(sites["resid_cross"])
        sites["mlp_out"] = self.fc2(F.gelu(self.fc1(sites["mlp_norm"])))
        sites["resid_post"] = sites["resid_cross"] + sites["mlp_out"]
        return sites["resid_post"]


class Encoder(nn.Module):
    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cfg.n_mels, cfg.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1)
        self.positions = nn.Parameter(torch.zeros(cfg.max_source_positions, cfg.d_model))
        self.layers = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_layers))
        self.final_norm = LayerNorm(cfg.d_model, cfg.layer_norm_eps)


class Decoder(nn.Module):
    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.positions = nn.Parameter(torch.zeros(cfg.max_target_positions, cfg.d_model))
        self.layers = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.final_norm = LayerNorm(cfg.d_model, cfg.layer_norm_eps)


class Whisper(nn.Module):
    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @torch.no_grad()
    def encode(self, features: Tensor, capture: Capture | None = None) -> CrossKv:
        """Runs the encoder over log-mel features [B, n_mels, 2 * max_source_positions].

        Returns the cross-attention keys and values of every decoder layer. If capture is given
        it is filled with every "encoder.*" site.
        """
        if features.shape[1:] != (self.cfg.n_mels, self.cfg.n_frames):
            raise ValueError(f"features {tuple(features.shape)} are not [B, n_mels, n_frames]")
        if capture is None:
            capture = {}
        enc = self.encoder
        # GELU is the exact erf form. Sites are frame-major: one row per frame.
        x = F.gelu(enc.conv1(features.to(enc.conv1.weight)))
        capture["encoder.conv1"] = x.transpose(1, 2)
        x = capture["encoder.conv2"] = F.gelu(enc.conv2(x)).transpose(1, 2)
        x = capture["encoder.embed"] = x + enc.positions
        for i, layer in enumerate(enc.layers):
            sites: Capture = {}
            x = layer(x, sites)
            capture.update({f"encoder.layers.{i}.{name}": v for name, v in sites.items()})
        x = capture["encoder.final_norm"] = enc.final_norm(x)

        blocks = cast(list[DecoderBlock], list(self.decoder.layers))
        cross = [block.cross_attn.kv(x) for block in blocks]
        for j, (k, v) in enumerate(cross):
            capture[f"encoder.cross_kv.{j}"] = torch.cat((k, v), dim=-1)
        return cross

    @torch.no_grad()
    def hidden(self, input_ids: Tensor, cross: CrossKv, capture: Capture | None = None) -> Tensor:
        """Runs the decoder over input_ids [B, T] from position 0; returns the final-norm output.

        If capture is given it is filled with every "decoder.*" site.
        """
        if capture is None:
            capture = {}
        dec = self.decoder
        x = capture["decoder.embed"] = dec.embed(input_ids) + dec.positions[: input_ids.shape[1]]
        for i, layer in enumerate(dec.layers):
            sites: Capture = {}
            x = layer(x, *cross[i], sites)
            capture.update({f"decoder.layers.{i}.{name}": v for name, v in sites.items()})
        capture["decoder.final_norm"] = dec.final_norm(x)
        return capture["decoder.final_norm"]

    def project(self, hidden: Tensor) -> Tensor:
        """Logits of final-norm rows: the output projection is the token embedding."""
        return F.linear(hidden, self.decoder.embed.weight)

    @torch.no_grad()
    def forward(
        self, features: Tensor, input_ids: Tensor, capture: Capture | None = None
    ) -> Tensor:
        """Teacher-forced forward; returns logits [B, T, vocab].

        If capture is given it is filled with every site of site_names.
        """
        if capture is None:
            capture = {}
        cross = self.encode(features, capture)
        capture["logits"] = self.project(self.hidden(input_ids, cross, capture))
        return capture["logits"]


@dataclass(frozen=True)
class WhisperRules:
    """Greedy decoding rules of a checkpoint, stored in its generation.json."""

    # Tokens fed before the first generated one, starting with the decoder start token.
    prompt: list[int]
    # Never generated.
    suppress_tokens: list[int]
    # Not generated as the first token after the prompt.
    begin_suppress_tokens: list[int]
    eos_token_id: int
    # Largest length of the whole sequence, prompt included.
    max_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def check(self, cfg: WhisperConfig) -> None:
        """Raises unless the rules fit the model."""
        ids = [*self.prompt, *self.suppress_tokens, *self.begin_suppress_tokens, self.eos_token_id]
        if not self.prompt or any(not 0 <= token < cfg.vocab_size for token in ids):
            raise ValueError(f"prompt is empty or a token id is outside the vocabulary: {self}")
        if not len(self.prompt) < self.max_length <= cfg.max_target_positions:
            raise ValueError(
                f"max_length {self.max_length} is not above the prompt length and within "
                f"max_target_positions {cfg.max_target_positions}"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WhisperRules":
        return cls(
            prompt=list(raw["prompt"]),
            suppress_tokens=list(raw["suppress_tokens"]),
            begin_suppress_tokens=list(raw["begin_suppress_tokens"]),
            eos_token_id=raw["eos_token_id"],
            max_length=raw["max_length"],
        )

    @classmethod
    def from_json(cls, path: Path) -> "WhisperRules":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@torch.no_grad()
def greedy(
    model: Whisper, features: Tensor, rules: WhisperRules, scores: list[Tensor] | None = None
) -> list[int]:
    """Greedy transcription of one window of features [n_mels, frames].

    Returns the whole sequence: the prompt, then the generated tokens, an end-of-sequence token
    included. The rules are those of Hugging Face `generate` without sampling or beams:
    suppress_tokens are never chosen, begin_suppress_tokens are not chosen as the first generated
    token, the lowest id wins a tie, and decoding stops after emitting eos or when the sequence
    holds max_length tokens.

    If scores is given, the logits [vocab] each generated token was chosen from are appended to
    it, suppressed entries at -inf.
    """
    rules.check(model.cfg)
    device = model.decoder.embed.weight.device
    cross = model.encode(features[None])
    ids = list(rules.prompt)
    while len(ids) < rules.max_length:
        # The whole prefix runs again each step: no cache, so no cache to get wrong.
        last = model.hidden(torch.tensor([ids], device=device), cross)[0, -1]
        logits = model.project(last).float()
        logits[rules.suppress_tokens] = float("-inf")
        if len(ids) == len(rules.prompt):
            logits[rules.begin_suppress_tokens] = float("-inf")
        if scores is not None:
            scores.append(logits.cpu())
        ids.append(int(logits.argmax()))
        if ids[-1] == rules.eos_token_id:
            break
    return ids


def top2_gap(scores: list[Tensor]) -> float:
    """Smallest margin by which a greedy step chose its token over the runner-up, in logits."""
    top = torch.stack(scores).topk(2, dim=-1).values
    return float((top[:, 0] - top[:, 1]).min())


def hf_weight_names(cfg: WhisperConfig) -> dict[str, str]:
    """Maps this module's parameter names to the tensor names in a converted checkpoint.

    These are the Hugging Face names plus a bias for every k_proj.
    """

    def affine(ours: str, theirs: str) -> dict[str, str]:
        return {f"{ours}.{part}": f"{theirs}.{part}" for part in ("weight", "bias")}

    def attention(ours: str, theirs: str) -> dict[str, str]:
        names: dict[str, str] = {}
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            names |= affine(f"{ours}.{proj}", f"{theirs}.{proj}")
        return names

    names = {"decoder.embed.weight": "model.decoder.embed_tokens.weight"}
    for side, n_layers in (("encoder", cfg.encoder_layers), ("decoder", cfg.decoder_layers)):
        names[f"{side}.positions"] = f"model.{side}.embed_positions.weight"
        names |= affine(f"{side}.final_norm", f"model.{side}.layer_norm")
        for i in range(n_layers):
            ours, theirs = f"{side}.layers.{i}", f"model.{side}.layers.{i}"
            names |= affine(f"{ours}.attn_norm", f"{theirs}.self_attn_layer_norm")
            names |= attention(f"{ours}.attn", f"{theirs}.self_attn")
            if side == "decoder":
                names |= affine(f"{ours}.cross_norm", f"{theirs}.encoder_attn_layer_norm")
                names |= attention(f"{ours}.cross_attn", f"{theirs}.encoder_attn")
            names |= affine(f"{ours}.mlp_norm", f"{theirs}.final_layer_norm")
            names |= affine(f"{ours}.fc1", f"{theirs}.fc1")
            names |= affine(f"{ours}.fc2", f"{theirs}.fc2")
    for conv in ("conv1", "conv2"):
        names |= affine(f"encoder.{conv}", f"model.encoder.{conv}")
    return names


def load_whisper(checkpoint_dir: Path, weights: WeightPrecision = "fp16") -> Whisper:
    """Builds the FP32 reference model from <dir>/config.json and <dir>/model.safetensors.

    A quantized checkpoint (kvbound.weight_format) is dequantized to the exact values the engine
    uses, whatever `weights` says.
    """
    checkpoint_dir = Path(checkpoint_dir)
    cfg = WhisperConfig.from_json(checkpoint_dir / "config.json")
    model = Whisper(cfg)
    load_weights(model, hf_weight_names(cfg), checkpoint_dir / "model.safetensors", weights)
    return model.eval()
