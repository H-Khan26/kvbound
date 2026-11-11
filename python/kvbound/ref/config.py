"""Architecture dimensions of a Llama-family checkpoint, read from its config.json."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
            "rope_scaling": raw.get("rope_scaling") is not None,
            "attention_bias": bool(raw.get("attention_bias", False)),
            "mlp_bias": bool(raw.get("mlp_bias", False)),
        }
        bad = [key for key, is_bad in unsupported.items() if is_bad]
        if bad:
            raise ValueError(f"unsupported config fields: {bad}")
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
        )
        if cfg.hidden_size % cfg.n_heads or cfg.n_heads % cfg.n_kv_heads or cfg.head_dim % 2:
            raise ValueError(f"inconsistent head dimensions: {cfg}")
        if raw.get("head_dim", cfg.head_dim) != cfg.head_dim:
            raise ValueError("head_dim differs from hidden_size / num_attention_heads")
        return cfg

    @classmethod
    def from_json(cls, path: Path) -> "LlamaConfig":
        return cls.from_hf_dict(json.loads(Path(path).read_text(encoding="utf-8")))
