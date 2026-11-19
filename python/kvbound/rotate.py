"""Residual-stream rotation before int4_g quantization (QuaRot's R1, Ashkboos et al., 2024; the
incoherence processing of QuIP, Chee et al., 2023).

An int4_g group's step is its largest weight over 7, so a group holding one outsized weight
rounds all the others coarsely. In a Llama the outsized values line up with a few channels of the
residual stream: the norm gains that scale those channels, the matrix columns that read them,
the activations that flow through them. An orthogonal change of basis of the residual stream
leaves the function the model computes unchanged, and a well-chosen one mixes every channel into
every other, so no single channel stands out in any group.

Conventions: a token is a row vector and a matrix W [out, in] computes y = x W^T, as nn.Linear
does. The rotated residual stream is h' = h Q for an orthogonal Q [hidden, hidden] (Q Q^T = I),
so h = h' Q^T. Every tensor that writes the stream or reads it absorbs Q:

  embedding E [vocab, hidden]   each row is a stream vector:            E -> E Q
  o_proj, down_proj  [hidden, n] write the stream, out' = out Q = z (Q^T W)^T:  W -> Q^T W
  q/k/v, gate/up, lm_head        read a norm of the stream:          W -> W diag(gamma) Q

The readers are the subtle part. RMSNorm(h) = h / rms(h) * gamma, and rms(h') = rms(h) because
Q keeps lengths (|h Q| = |h|, and rms is |h| / sqrt(hidden)). The gain gamma scales each channel
on its own, which does not commute with Q, so it is folded into the matrices that read the norm
first (q/k/v after input_layernorm, gate/up after post_attention_layernorm, lm_head after the
final norm) and set to ones. Then

  RMSNorm'(h') W'^T = (h' / rms(h)) Q^T diag(gamma) W^T = (h / rms(h)) diag(gamma) W^T
                    = RMSNorm(h) W^T

Nothing inside attention or the MLP moves: q, k, v and their RoPE, the softmax, and the MLP
hidden that down_proj reads are as before. In particular down_proj's input is not the stream,
so a massive activation in it stays in its one channel.

Tied embeddings: after folding, the head is E diag(gamma_final) Q and the embedding E Q, so one
tensor no longer serves both. The rotated checkpoint stores lm_head and its config.json says
tie_word_embeddings false, which the engine and the reference model read as for any untied
model. The cost is vocab x hidden more stored weights (Llama 3.2 1B: 128256 x 2048, 263M, a
fifth more); decoding reads the head once per step either way and the embedding one row per
token, so the bytes a decode step streams do not change.

Q is a randomized Hadamard matrix, Q = diag(s) H / sqrt(hidden): H is the Sylvester Hadamard
matrix (H_1 = [1], H_2n = [[H_n, H_n], [H_n, -H_n]]), whose entries are all +-1 and whose rows
are orthogonal, and s holds seeded random signs. Every entry of Q has magnitude 1 / sqrt(hidden),
so each rotated channel is an equal mix of all the original ones, and energy in one channel
spreads evenly over all of them; the signs keep a structured input (a constant vector, say) from
landing on a single row of H. The hidden size must be a power of two.

The algebra runs in float64 on the stored FP16 weights, and each result is rounded to FP16 once.
The output is a plain FP16 checkpoint (variant "fp16.rot") that computes what its source does,
and that kvbound.quantize turns into int4_g by rounding to nearest or by GPTQ like any other.

AWQ does not compose with it as built: AWQ's scales live in the norm gains, which the rotation
folds away, and a per-channel scale of the rotated stream is not one of the original stream.

Usage: python -m kvbound.rotate llama-3.2-1b [--seed 0]
       python -m kvbound.quantize llama-3.2-1b --scheme int4_g --rotate [--gptq]
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from kvbound.models import load_spec, sha256_file, write_json
from kvbound.quantize import CHUNK_ELEMENTS, COPIED_FILES, WEIGHTS_FILE, to_f16
from kvbound.ref.config import LlamaConfig
from kvbound.weight_format import (
    FP16,
    HADAMARD,
    KEY_ROTATION_SEED,
    KEY_SOURCE,
    KEY_SOURCE_SHA256,
    WeightFormat,
)

ROTATE_MANIFEST_FILE = "rotate_manifest.json"
EMBED = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
FINAL_NORM = "model.norm.weight"
# Per matrix that reads a norm of the stream, the norm (within its layer for layer tensors).
READ_NORMS = {
    "self_attn.q_proj.weight": "input_layernorm.weight",
    "self_attn.k_proj.weight": "input_layernorm.weight",
    "self_attn.v_proj.weight": "input_layernorm.weight",
    "mlp.gate_proj.weight": "post_attention_layernorm.weight",
    "mlp.up_proj.weight": "post_attention_layernorm.weight",
}
WRITERS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")


def hadamard(n: int) -> Tensor:
    """The Sylvester Hadamard matrix of order n (a power of two), float64 entries +-1."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"a Sylvester Hadamard matrix needs a power-of-two order, got {n}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)))
    return h


def random_hadamard(n: int, seed: int) -> Tensor:
    """Q = diag(s) H / sqrt(n), float64 [n, n], with seeded random signs s."""
    gen = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=gen).double() * 2 - 1
    return signs[:, None] * hadamard(n) / math.sqrt(n)


def rotate_input(w: Tensor, gain: Tensor | None, q: Tensor) -> Tensor:
    """W diag(gain) Q in float64 for a matrix w [out, hidden] that reads the stream (or the
    embedding, with no gain), a block of rows at a time."""
    rows = max(1, CHUNK_ELEMENTS // w.shape[1])
    out = torch.empty(w.shape, dtype=torch.float64)
    for start in range(0, w.shape[0], rows):
        chunk = w[start : start + rows].double()
        if gain is not None:
            chunk = chunk * gain
        out[start : start + rows] = chunk @ q
    return out


def rotate_output(w: Tensor, q: Tensor) -> Tensor:
    """Q^T W in float64 for a matrix w [hidden, n] that writes the stream."""
    return q.T @ w.double()


def layer_part(name: str) -> tuple[str, str] | None:
    """("model.layers.<i>.", rest of the name) for a layer tensor, else None."""
    parts = name.split(".", 3)
    if len(parts) == 4 and parts[0] == "model" and parts[1] == "layers":
        return f"model.layers.{parts[2]}.", parts[3]
    return None


def rotate_tensors(tensors: dict[str, Tensor], cfg: LlamaConfig, q: Tensor) -> dict[str, Tensor]:
    """The rotated checkpoint's tensors, FP16, from a Llama's plain ones (under Hugging Face
    names): the stream rotated by q, the norm gains folded in and set to ones, and lm_head stored
    even when cfg ties it. A tied checkpoint's duplicate lm_head, if stored, is ignored."""
    hidden = cfg.hidden_size
    if q.shape != (hidden, hidden):
        raise ValueError(f"the rotation is {tuple(q.shape)}, the stream {hidden} wide")
    out: dict[str, Tensor] = {}
    head_source = EMBED if cfg.tie_word_embeddings else LM_HEAD
    out[LM_HEAD] = to_f16(
        rotate_input(tensors[head_source], tensors[FINAL_NORM].double(), q), LM_HEAD
    )
    for name, w in tensors.items():
        if name == LM_HEAD:
            continue
        if name == EMBED:
            rotated = rotate_input(w, None, q)
        elif name == FINAL_NORM:
            rotated = torch.ones_like(w, dtype=torch.float64)
        else:
            split = layer_part(name)
            if split is None:
                raise ValueError(f"{name}: not a tensor of a Llama this rotation knows")
            layer, rest = split
            if rest in READ_NORMS:
                rotated = rotate_input(w, tensors[layer + READ_NORMS[rest]].double(), q)
            elif rest in WRITERS:
                rotated = rotate_output(w, q)
            elif rest in NORMS:
                rotated = torch.ones_like(w, dtype=torch.float64)
            else:
                raise ValueError(f"{name}: not a tensor of a Llama this rotation knows")
        out[name] = to_f16(rotated, name)
    return out


def rotate_checkpoint(
    src_dir: Path, out_dir: Path, seed: int = 0, source: str = ""
) -> dict[str, Any]:
    """Writes the rotated copy of the plain Llama checkpoint in src_dir to out_dir as FP16;
    returns its manifest, also written to rotate_manifest.json.

    The header holds kvbound.rotation, the seed of the signs and the source's name and hash;
    config.json is the source's with tie_word_embeddings false.
    """
    started = time.perf_counter()
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    cfg = LlamaConfig.from_hf_dict(config)
    src = src_dir / WEIGHTS_FILE
    with safe_open(str(src), framework="pt") as f:
        src_fmt = WeightFormat.from_metadata(f.metadata())
        if src_fmt.quantized or src_fmt.rotation:
            raise ValueError(f"{src} is not a plain checkpoint ({src_fmt.variant})")
        tensors = {name: f.get_tensor(name) for name in list(f.keys())}
    q = random_hadamard(cfg.hidden_size, seed)
    rotated = rotate_tensors(tensors, cfg, q)
    del tensors

    fmt = WeightFormat.make(FP16, rotation=HADAMARD)
    source_sha256 = sha256_file(src)
    metadata = fmt.to_metadata() | {KEY_ROTATION_SEED: str(seed), KEY_SOURCE_SHA256: source_sha256}
    if source:
        metadata[KEY_SOURCE] = source
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(rotated, str(out_dir / WEIGHTS_FILE), metadata=metadata)
    write_json(out_dir / "config.json", config | {"tie_word_embeddings": False})
    for file in COPIED_FILES:
        if file != "config.json" and (src_dir / file).exists():
            shutil.copyfile(src_dir / file, out_dir / file)

    manifest = {
        "format": metadata,
        "variant": fmt.variant,
        "rotation": HADAMARD,
        "seed": seed,
        "hidden_size": cfg.hidden_size,
        "source_tied": cfg.tie_word_embeddings,
        # Weights stored beyond the source's: the head a tied source shared with the embedding.
        "added_params": rotated[LM_HEAD].numel() if cfg.tie_word_embeddings else 0,
        "params": sum(t.numel() for t in rotated.values()),
        "file_bytes": (out_dir / WEIGHTS_FILE).stat().st_size,
        "source_sha256": source_sha256,
        "seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
    }
    write_json(out_dir / ROTATE_MANIFEST_FILE, manifest)
    return manifest


def rotation_seed(model_dir: Path) -> int | None:
    """The seed of a rotated checkpoint's signs; None for a file that is not rotated."""
    with safe_open(str(Path(model_dir) / WEIGHTS_FILE), framework="pt") as f:
        metadata = f.metadata() or {}
    if not WeightFormat.from_metadata(metadata).rotation:
        return None
    return int(metadata[KEY_ROTATION_SEED])


def ensure_rotated(name: str, seed: int) -> Path:
    """The rotated FP16 checkpoint of registry model `name` with this seed, written first when
    it does not exist yet."""
    spec = load_spec(name)
    if spec.family != "llama":
        raise ValueError(f"the rotation applies to a Llama; {spec.name} is a {spec.family}")
    out_dir = spec.variant_dir(WeightFormat.make(FP16, rotation=HADAMARD).variant)
    if (out_dir / WEIGHTS_FILE).exists():
        found = rotation_seed(out_dir)
        if found != seed:
            raise ValueError(f"{out_dir} holds the rotation of seed {found}, not {seed}")
        return out_dir
    rotate_checkpoint(spec.checkpoint_dir, out_dir, seed, f"{spec.hf_repo}@{spec.revision}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random signs")
    args = parser.parse_args()
    started = time.perf_counter()
    out_dir = ensure_rotated(args.model, args.seed)
    print(f"{out_dir} ({time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()
