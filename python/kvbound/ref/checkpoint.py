"""Loads a safetensors checkpoint into a reference model's parameters."""

from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from torch import nn

from kvbound.weight_format import WeightFormat

# "native" keeps the stored values; "fp16" rounds them to the engine's weight precision first.
WeightPrecision = Literal["native", "fp16"]


def load_weights(
    model: nn.Module,
    names: dict[str, str],
    weights_file: Path,
    weights: WeightPrecision,
    ignored: frozenset[str] = frozenset(),
) -> None:
    """Copies the tensors of weights_file into model's parameters as float32.

    names maps each parameter name to its tensor name in the file. The file must hold exactly
    those tensors, apart from the names in `ignored`. A quantized checkpoint
    (kvbound.weight_format) is dequantized to the exact values the engine uses, whatever
    `weights` says.
    """
    params = dict(model.named_parameters())  # tied parameters are listed once
    with safe_open(str(weights_file), framework="pt") as f:
        fmt = WeightFormat.from_metadata(f.metadata())
        stored = set(f.keys())
        if fmt.quantized:
            # Imported here so plain checkpoints do not load the converter and model registry.
            from kvbound.quantize import read_tensor, stored_names  # noqa: PLC0415

            stored = set(stored_names(list(stored), fmt))
        unexpected = stored - set(names.values()) - ignored
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
