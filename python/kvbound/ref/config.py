"""Architecture dimensions of a Llama or Whisper checkpoint, read from its config.json."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Llama3RopeScaling:
    """Llama 3 RoPE frequency scaling: slows the low-frequency rotations by `factor`."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_scaling: Llama3RopeScaling | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def n_rep(self) -> int:
        """Query heads served by each KV head (grouped-query attention)."""
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_hf_dict(cls, raw: dict[str, Any]) -> "LlamaConfig":
        # Refuse variants this implementation does not model rather than run them wrongly.
        unsupported = {
            "model_type": raw.get("model_type") != "llama",
            "hidden_act": raw.get("hidden_act", "silu") != "silu",
            "attention_bias": bool(raw.get("attention_bias", False)),
            "mlp_bias": bool(raw.get("mlp_bias", False)),
        }
        bad = [key for key, is_bad in unsupported.items() if is_bad]
        if bad:
            raise ValueError(f"unsupported config fields: {bad}")
        scaling = raw.get("rope_scaling")
        if scaling is not None:
            if scaling.get("rope_type") != "llama3":
                raise ValueError(f"unsupported rope_scaling: {scaling}")
            scaling = Llama3RopeScaling(
                factor=scaling["factor"],
                low_freq_factor=scaling["low_freq_factor"],
                high_freq_factor=scaling["high_freq_factor"],
                original_max_position_embeddings=scaling["original_max_position_embeddings"],
            )
        cfg = cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            n_layers=raw["num_hidden_layers"],
            n_heads=raw["num_attention_heads"],
            n_kv_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
            intermediate_size=raw["intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 10000.0),
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            rope_scaling=scaling,
        )
        if cfg.hidden_size % cfg.n_heads or cfg.n_heads % cfg.n_kv_heads or cfg.head_dim % 2:
            raise ValueError(f"inconsistent head dimensions: {cfg}")
        if raw.get("head_dim", cfg.head_dim) != cfg.head_dim:
            raise ValueError("head_dim differs from hidden_size / num_attention_heads")
        return cfg

    @classmethod
    def from_json(cls, path: Path) -> "LlamaConfig":
        return cls.from_hf_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class WhisperConfig:
    vocab_size: int
    d_model: int
    n_mels: int
    encoder_layers: int
    encoder_heads: int
    encoder_ffn_dim: int
    decoder_layers: int
    decoder_heads: int
    decoder_ffn_dim: int
    # Encoder positions. The two-stride conv stem halves the input, so it takes twice as many
    # mel frames.
    max_source_positions: int
    max_target_positions: int
    layer_norm_eps: float = 1e-5

    @property
    def n_frames(self) -> int:
        """Mel frames of one input window."""
        return 2 * self.max_source_positions

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_hf_dict(cls, raw: dict[str, Any]) -> "WhisperConfig":
        # Refuse variants this implementation does not model rather than run them wrongly.
        unsupported = {
            "model_type": raw.get("model_type") != "whisper",
            "activation_function": raw.get("activation_function", "gelu") != "gelu",
            "scale_embedding": bool(raw.get("scale_embedding", False)),
            "tie_word_embeddings": not raw.get("tie_word_embeddings", True),
        }
        bad = [key for key, is_bad in unsupported.items() if is_bad]
        if bad:
            raise ValueError(f"unsupported config fields: {bad}")
        cfg = cls(
            vocab_size=raw["vocab_size"],
            d_model=raw["d_model"],
            n_mels=raw["num_mel_bins"],
            encoder_layers=raw["encoder_layers"],
            encoder_heads=raw["encoder_attention_heads"],
            encoder_ffn_dim=raw["encoder_ffn_dim"],
            decoder_layers=raw["decoder_layers"],
            decoder_heads=raw["decoder_attention_heads"],
            decoder_ffn_dim=raw["decoder_ffn_dim"],
            max_source_positions=raw["max_source_positions"],
            max_target_positions=raw["max_target_positions"],
        )
        if cfg.d_model % cfg.encoder_heads or cfg.d_model % cfg.decoder_heads:
            raise ValueError(f"inconsistent head dimensions: {cfg}")
        return cfg

    @classmethod
    def from_json(cls, path: Path) -> "WhisperConfig":
        return cls.from_hf_dict(json.loads(Path(path).read_text(encoding="utf-8")))
