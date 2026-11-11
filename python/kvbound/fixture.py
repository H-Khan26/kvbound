"""Generates fixtures/tiny-llama: a random 2-layer Llama small enough to commit, plus its goldens.

It mirrors the primary model's variants: grouped-query attention, tied embeddings and Llama 3
RoPE scaling.

CI has no GPU and no real weights, so every CPU test (Python and C++) runs against this fixture.
It has the same file layout as a real checkpoint (config.json + model.safetensors with Hugging
Face tensor names), so the same loaders are exercised.

Usage: python -m kvbound.fixture
"""

import json

import torch
from safetensors.torch import save_file

from kvbound import golden
from kvbound.models import sha256_file
from kvbound.paths import FIXTURES
from kvbound.ref import Llama, LlamaConfig, load_llama
from kvbound.ref.model import hf_weight_names

FIXTURE_DIR = FIXTURES / "tiny-llama"
SEED = 0
PROMPT_LENGTHS = (1, 5, 12)

HF_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_act": "silu",
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    # Llama 3 frequency scaling with all three bands in play for head_dim 16.
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 512,
    },
    "tie_word_embeddings": True,
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def random_weights(cfg: LlamaConfig, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """FP16 weights under Hugging Face names, scaled so activations stay O(1) through the stack."""
    shapes = {name: p.shape for name, p in Llama(cfg).named_parameters()}
    weights = {}
    for ours, theirs in hf_weight_names(cfg).items():
        shape = shapes[ours]
        if "norm" in ours:
            w = 1.0 + 0.1 * torch.randn(shape, generator=gen)
        elif ours == "embed.weight":
            w = torch.randn(shape, generator=gen)
        else:
            w = torch.randn(shape, generator=gen) / shape[1] ** 0.5
        weights[theirs] = w.to(torch.float16)
    return weights


def build() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(SEED)
    cfg = LlamaConfig.from_hf_dict(HF_CONFIG)

    with (FIXTURE_DIR / "config.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(HF_CONFIG, indent=2) + "\n")
    weights_path = FIXTURE_DIR / "model.safetensors"
    save_file(random_weights(cfg, gen), str(weights_path), metadata={"format": "pt"})

    prompts = [
        golden.Prompt(f"p{i}", torch.randint(cfg.vocab_size, (n,), generator=gen).tolist())
        for i, n in enumerate(PROMPT_LENGTHS)
    ]
    meta = {
        "name": "tiny-llama",
        "weights_sha256": sha256_file(weights_path),
        "weights_precision": "fp16",
        "seed": SEED,
    }
    # Goldens come from the file as stored, through the same loader real checkpoints use.
    golden.dump(load_llama(FIXTURE_DIR), prompts, FIXTURE_DIR, meta)


if __name__ == "__main__":
    build()
    print(FIXTURE_DIR)
