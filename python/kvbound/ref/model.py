"""From-scratch Llama reference model: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP.

This is the ground truth the engine is tested against. It uses no Hugging Face modeling code and
loads weights straight from a safetensors file. Every intermediate the engine must reproduce can
be captured by name (see SITES).
"""

import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from torch import Tensor, nn

from kvbound.ref.config import Llama3RopeScaling, LlamaConfig
from kvbound.weight_format import WeightFormat

# Capture sites, in forward order. "layers.{i}.*" repeats per layer.
LAYER_SITES = ("attn_norm", "attn_out", "resid_mid", "mlp_norm", "mlp_out", "resid_post")
SITES = ("embed", *(f"layers.{{i}}.{s}" for s in LAYER_SITES), "final_norm", "logits")

# "native" keeps the stored values; "fp16" rounds them to the engine's weight precision first.
WeightPrecision = Literal["native", "fp16"]

Capture = dict[str, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def llama3_scale(inv_freq: Tensor, s: Llama3RopeScaling) -> Tensor:
    """Llama 3 frequency scaling, by wavelength (positions per full rotation).

    Wavelengths shorter than original_max / high_freq_factor keep their frequency, those longer
    than original_max / low_freq_factor are slowed by `factor`, and the band between blends
    linearly from one to the other.
    """
    wavelen = 2 * math.pi / inv_freq
    low_freq_wavelen = s.original_max_position_embeddings / s.low_freq_factor
    high_freq_wavelen = s.original_max_position_embeddings / s.high_freq_factor
    smooth = (s.original_max_position_embeddings / wavelen - s.low_freq_factor) / (
        s.high_freq_factor - s.low_freq_factor
    )
    blended = (1 - smooth) * inv_freq / s.factor + smooth * inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / s.factor, inv_freq)
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, blended, scaled)


def rope_tables(
    head_dim: int, theta: float, positions: Tensor, scaling: Llama3RopeScaling | None = None
) -> tuple[Tensor, Tensor]:
    """cos and sin of position * inv_freq_j, shape [T, head_dim/2].

    inv_freq_j = theta^(-2j/head_dim), optionally rescaled the Llama 3 way.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (theta ** (exponents / head_dim))
    if scaling is not None:
        inv_freq = llama3_scale(inv_freq, scaling)
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates each (j, j + head_dim/2) pair of x [B, heads, T, head_dim] by its position angle."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = cfg.n_rep
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Grouped-query attention: KV head j serves query heads [j * n_rep, (j + 1) * n_rep).
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        # Softmax runs in FP32 whatever the model precision, as the engine's kernel does.
        probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
        out = (probs @ v).transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, sites: Capture) -> Tensor:
        sites["attn_norm"] = self.attn_norm(x)
        sites["attn_out"] = self.attn(sites["attn_norm"], cos, sin)
        sites["resid_mid"] = x + sites["attn_out"]
        sites["mlp_norm"] = self.mlp_norm(sites["resid_mid"])
        sites["mlp_out"] = self.mlp(sites["mlp_norm"])
        sites["resid_post"] = sites["resid_mid"] + sites["mlp_out"]
        return sites["resid_post"]


class Llama(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    @torch.no_grad()
    def forward(self, input_ids: Tensor, capture: Capture | None = None) -> Tensor:
        """Full-sequence forward over input_ids [B, T]; returns logits [B, T, vocab].

        If capture is given it is filled with every site in SITES.
        """
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        cfg = self.cfg
        cos, sin = rope_tables(cfg.head_dim, cfg.rope_theta, positions, cfg.rope_scaling)
        if capture is None:
            capture = {}

        x = capture["embed"] = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            sites: Capture = {}
            x = layer(x, cos, sin, sites)
            capture.update({f"layers.{i}.{name}": value for name, value in sites.items()})
        capture["final_norm"] = self.final_norm(x)
        capture["logits"] = self.lm_head(capture["final_norm"])
        return capture["logits"]


def hf_weight_names(cfg: LlamaConfig) -> dict[str, str]:
    """Maps this module's parameter names to the tensor names in a Hugging Face checkpoint."""
    names = {"embed.weight": "model.embed_tokens.weight", "final_norm.weight": "model.norm.weight"}
    if not cfg.tie_word_embeddings:
        names["lm_head.weight"] = "lm_head.weight"
    for i in range(cfg.n_layers):
        ours, theirs = f"layers.{i}", f"model.layers.{i}"
        names[f"{ours}.attn_norm.weight"] = f"{theirs}.input_layernorm.weight"
        names[f"{ours}.mlp_norm.weight"] = f"{theirs}.post_attention_layernorm.weight"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names[f"{ours}.attn.{proj}.weight"] = f"{theirs}.self_attn.{proj}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names[f"{ours}.mlp.{proj}.weight"] = f"{theirs}.mlp.{proj}.weight"
    return names


def load_llama(checkpoint_dir: Path, weights: WeightPrecision = "fp16") -> Llama:
    """Builds the FP32 reference model from <dir>/config.json and <dir>/model.safetensors.

    A quantized checkpoint (kvbound.weight_format) is dequantized to the exact values the engine
    uses, whatever `weights` says.
    """
    checkpoint_dir = Path(checkpoint_dir)
    cfg = LlamaConfig.from_json(checkpoint_dir / "config.json")
    model = Llama(cfg)
    names = hf_weight_names(cfg)
    params = dict(model.named_parameters())  # a tied lm_head is listed once, as embed.weight

    with safe_open(str(checkpoint_dir / "model.safetensors"), framework="pt") as f:
        fmt = WeightFormat.from_metadata(f.metadata())
        stored = set(f.keys())
        if fmt.quantized:
            # Imported here so plain checkpoints do not load the converter and model registry.
            from kvbound.quantize import read_tensor, stored_names  # noqa: PLC0415

            stored = set(stored_names(list(stored), fmt))
        # A tied checkpoint may still store the duplicate lm_head tensor.
        unexpected = stored - set(names.values()) - {"lm_head.weight"}
        missing = set(names.values()) - stored
        if unexpected or missing or set(params) != set(names):
            raise ValueError(f"checkpoint mismatch: unexpected={unexpected}, missing={missing}")
        with torch.no_grad():
            for ours, theirs in names.items():
                tensor = read_tensor(f, theirs, fmt) if fmt.quantized else f.get_tensor(theirs)
                if tensor.shape != params[ours].shape:
                    raise ValueError(f"{theirs}: shape {tuple(tensor.shape)} does not fit {ours}")
                if weights == "fp16" and not fmt.quantized:
                    tensor = tensor.to(torch.float16)
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"{theirs}: values overflow FP16")
                params[ours].copy_(tensor.to(torch.float32))
    return model.eval()
