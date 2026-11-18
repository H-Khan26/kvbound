"""faster-whisper speech baseline: the CTranslate2 Whisper runtime, as the Sawt Sakinah app runs it.

The fetched checkpoint is converted once to data/models/<name>.ct2 with FP16 weights, next to
the tokenizer.json of the converted checkpoint. CTranslate2 casts the weights to the compute
type when it loads the model: float16 and int8_float16 on the GPU, int8 on the CPU (the app's
setting; activations are then FP32).

Each clip goes through WhisperModel.transcribe with language "ar", no timestamps, temperature 0,
no previous-text prompt, no voice-activity filter and no thresholds that drop or redo a
transcript, so the text is one plain decode of one window. Two things differ from the Hugging
Face baseline and are part of this stack:

  prompt     faster-whisper forces <|startoftranscript|><|ar|><|transcribe|><|notimestamps|>;
             Hugging Face feeds the Tarteel checkpoints the start token alone.
  front end  the log-mel transform runs on the clip alone, clamped to 8 below the clip's own
             peak, and the features are then padded with zeros to 30 s. Hugging Face pads the
             samples with silence first.

The timings come from inside transcribe(): the CTranslate2 model is wrapped so that its encode()
and generate() calls are timed. encoder_ms is encode(), which takes the features from the host
and returns after its work on the device is done; the cross-attention keys and values are
projected inside generate(). generate() returns all tokens at once, so there is no first-token
time and no decode_ms_per_token.

Needs the optional group: uv sync --locked --extra cu128 --group faster-whisper
Usage: python -m kvbound.evals.asr eval --model whisper-base-ar-quran --backend faster-whisper
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

# Before CTranslate2: on Windows, importing torch loads the cuBLAS and cuDNN 9 libraries it ships.
# CTranslate2 ships only a cuDNN loader of another version and finds the rest by name.
import torch  # noqa: F401

# isort: split
import ctranslate2
import faster_whisper
from ctranslate2.converters import TransformersConverter

from kvbound.evals.asr import Transcript
from kvbound.models import ModelSpec

# Weight and activation precision of each compute type, in the record's vocabulary.
COMPUTE_TYPES = {
    "float16": ("fp16", "fp16"),
    "int8_float16": ("int8", "fp16"),
    "int8": ("int8", "fp32"),
    "float32": ("fp32", "fp32"),
}
DEFAULT_COMPUTE_TYPE = {"cuda": "float16", "cpu": "int8"}
LANGUAGE = "ar"
# Arguments of WhisperModel.transcribe besides the beam size.
OPTIONS: dict[str, Any] = {
    "language": LANGUAGE,
    "task": "transcribe",
    "without_timestamps": True,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "vad_filter": False,
    "word_timestamps": False,
    # Without these, a transcript is never dropped as silence or flagged for another attempt.
    "no_speech_threshold": None,
    "log_prob_threshold": None,
    "compression_ratio_threshold": None,
}


def convert(spec: ModelSpec) -> Path:
    """data/models/<name>.ct2, converted from the fetched checkpoint when it is not there."""
    out = spec.variant_dir("ct2")
    if not (out / "model.bin").exists():
        if not (spec.source_dir / "config.json").exists():
            raise FileNotFoundError(
                f"{spec.source_dir} not found; run `make fetch MODEL={spec.name}`"
            )
        TransformersConverter(str(spec.source_dir)).convert(
            str(out), quantization="float16", force=True
        )
        # The source repository may hold no tokenizer.json; without one faster-whisper would
        # download the tokenizer of openai/whisper-tiny.
        shutil.copyfile(spec.checkpoint_dir / "tokenizer.json", out / "tokenizer.json")
        preprocessor = spec.source_dir / "preprocessor_config.json"
        if preprocessor.exists():
            shutil.copyfile(preprocessor, out / preprocessor.name)
    return out


class TimedModel:
    """Stands in for the CTranslate2 model inside a WhisperModel and times its calls."""

    def __init__(self, model: ctranslate2.models.Whisper) -> None:
        self._model = model
        self.encode_s: list[float] = []
        self.generate_s: list[float] = []
        self.prompts: list[list[int]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        out = self._model.encode(*args, **kwargs)
        self.encode_s.append(time.perf_counter() - started)
        return out

    def generate(self, features: Any, prompts: list[list[int]], **kwargs: Any) -> Any:
        started = time.perf_counter()
        out = self._model.generate(features, prompts, **kwargs)
        self.generate_s.append(time.perf_counter() - started)
        self.prompts += [list(prompt) for prompt in prompts]
        return out


class FasterWhisper:
    """A converted Whisper model behind kvbound.evals.asr.Transcriber."""

    def __init__(self, spec: ModelSpec, device: str, compute_type: str, beams: int) -> None:
        self.model = faster_whisper.WhisperModel(
            str(convert(spec)), device=device, compute_type=compute_type, local_files_only=True
        )
        self.timed = TimedModel(self.model.model)
        self.model.model = self.timed
        self.eos = self.model.hf_tokenizer.token_to_id("<|endoftext|>")
        self.options = {**OPTIONS, "beam_size": beams}

        weights, kv = COMPUTE_TYPES[compute_type]
        self.engine = {
            "name": "faster-whisper",
            "version": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__,
        }
        self.params = {
            "backend": "faster-whisper",
            "device": device,
            "compute_type": compute_type,
            "weights": weights,
            "kv": kv,
            "beams": beams,
            "transcribe_options": OPTIONS,
            "encoder_ms_covers": (
                "encode(): features on the host to the encoder output on the device; the "
                "cross-attention keys and values are projected inside generate()"
            ),
        }
        if device == "cpu":
            # CTranslate2 takes its thread count from OMP_NUM_THREADS, and uses 4 without it.
            self.params["cpu_threads"] = int(os.environ.get("OMP_NUM_THREADS", "4"))

    def __call__(self, samples: np.ndarray) -> Transcript:
        timed = self.timed
        timed.encode_s.clear()
        timed.generate_s.clear()
        timed.prompts.clear()
        started = time.perf_counter()
        segments, _ = self.model.transcribe(samples, **self.options)
        # The segments are decoded as the generator is consumed.
        segments = list(segments)
        e2e_s = time.perf_counter() - started
        if len(timed.encode_s) != 1 or len(timed.generate_s) != 1:
            raise ValueError(
                f"the clip took {len(timed.encode_s)} encoder and {len(timed.generate_s)} "
                "decoder calls; expected one window"
            )
        tokens = [token for segment in segments for token in segment.tokens if token != self.eos]
        return Transcript(
            text="".join(segment.text for segment in segments).strip(),
            prompt=timed.prompts[0],
            tokens=tokens,
            e2e_s=e2e_s,
            model_s=timed.encode_s[0] + timed.generate_s[0],
            encoder_s=timed.encode_s[0],
        )


def load(spec: ModelSpec, device: str, compute_type: str | None, beams: int) -> FasterWhisper:
    """The converted model of `spec` on `device`; compute_type is a CTranslate2 compute type."""
    compute_type = compute_type or DEFAULT_COMPUTE_TYPE[device]
    if compute_type not in COMPUTE_TYPES:
        raise ValueError(f"unknown compute type {compute_type!r}; known: {sorted(COMPUTE_TYPES)}")
    return FasterWhisper(spec, device, compute_type, beams)
