"""Hugging Face Transformers speech baseline: Whisper through `generate`, one 30 s window a clip.

The model is the checkpoint as downloaded, FP16 on the GPU and FP32 on the CPU, decoded without
sampling or timestamps, greedily or with beams. The front end is kvbound.audio, which pads the
samples to 30 s before the log-mel transform, as Hugging Face and OpenAI do.

`generate` picks the prompt from the checkpoint's generation settings. It is read off the first
decoding step and must be the prompt in the model's toml, so a record always says what the
decoder was fed: the Tarteel checkpoints get the start token alone and emit their own language
and task tokens, which count as generated tokens here.

encoder_ms is the forward pass of the encoder module, with the features already on the device;
the cross-attention keys and values are projected in the first decoder step. The time of each
decoder step comes from a logits processor that runs last and changes nothing.

Usage: python -m kvbound.evals.asr eval --model whisper-base-ar-quran --backend hf
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import transformers
from tokenizers import Tokenizer
from transformers import LogitsProcessor, WhisperForConditionalGeneration

from kvbound import audio
from kvbound.evals.asr import Transcript
from kvbound.models import ModelSpec
from kvbound.ref import WhisperRules

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
DEFAULT_COMPUTE_TYPE = {"cuda": "fp16", "cpu": "fp32"}


class StepTimer(LogitsProcessor):
    """Notes when the scores of each decoding step are ready, and what the first step was fed.

    It runs after the model's own processors and returns the scores unchanged.
    """

    def __init__(self, sync: Callable[[], None]) -> None:
        self.sync = sync
        self.times: list[float] = []
        self.prompt: list[int] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.sync()
        self.times.append(time.perf_counter())
        if len(self.times) == 1:
            # Every beam starts from the same ids.
            self.prompt = input_ids[0].tolist()
        return scores


class HfWhisper:
    """A Whisper checkpoint behind kvbound.evals.asr.Transcriber."""

    def __init__(self, spec: ModelSpec, device: str, compute_type: str, beams: int) -> None:
        self.rules = WhisperRules.from_dict(spec.generation)
        self.device, self.dtype, self.beams = device, DTYPES[compute_type], beams
        # Any: the transformers type stubs reject valid generate() and to() calls.
        model: Any = WhisperForConditionalGeneration.from_pretrained(
            spec.source_dir, dtype=self.dtype
        )
        self.model = model.to(device).eval()
        self.tokenizer = Tokenizer.from_file(str(spec.checkpoint_dir / "tokenizer.json"))
        # A prompt longer than the start token names the language and the task.
        names = [
            self.tokenizer.decode([token], False).strip("<|>") for token in self.rules.prompt[1:]
        ]
        self.generate_kwargs = {"language": names[0], "task": names[1]} if names else {}
        self.sync = torch.cuda.synchronize if device == "cuda" else lambda: None
        # The encoder runs once inside generate(); these hooks time it.
        self.encoder_times: list[float] = []
        encoder = self.model.get_encoder()
        encoder.register_forward_pre_hook(lambda *_: self.stamp_encoder())
        encoder.register_forward_hook(lambda *_: self.stamp_encoder())

        self.engine = {"name": "hf-transformers", "version": transformers.__version__}
        self.params = {
            "backend": "hf",
            "device": device,
            "compute_type": compute_type,
            "weights": compute_type,
            "kv": compute_type,
            "beams": beams,
            "encoder_ms_covers": (
                "forward pass of the encoder module, features already on the device; the "
                "cross-attention keys and values are projected in the first decoder step"
            ),
        }
        if device == "cpu":
            self.params["cpu_threads"] = torch.get_num_threads()

    def stamp_encoder(self) -> None:
        self.sync()
        self.encoder_times.append(time.perf_counter())

    @torch.no_grad()
    def __call__(self, samples: np.ndarray) -> Transcript:
        self.sync()
        started = time.perf_counter()
        features = audio.features(samples, self.model.config.num_mel_bins)
        featured = time.perf_counter()
        self.encoder_times.clear()
        timer = StepTimer(self.sync)
        # As a dict, the output keeps the prompt and the end token in the sequence.
        out = self.model.generate(
            features[None].to(self.device, self.dtype),
            do_sample=False,
            num_beams=self.beams,
            logits_processor=[timer],
            return_dict_in_generate=True,
            **self.generate_kwargs,
        )
        ids = out.sequences[0].tolist()
        decoded = time.perf_counter()

        if timer.prompt != self.rules.prompt or ids[: len(timer.prompt)] != timer.prompt:
            raise ValueError(f"generate() fed the prompt {timer.prompt}, not {self.rules.prompt}")
        if len(self.encoder_times) != 2:
            raise ValueError(f"the encoder ran {len(self.encoder_times) // 2} times for one clip")
        tokens = ids[len(timer.prompt) :]
        while tokens and tokens[-1] == self.rules.eos_token_id:
            tokens.pop()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return Transcript(
            text=text,
            prompt=timer.prompt,
            tokens=tokens,
            e2e_s=time.perf_counter() - started,
            model_s=decoded - featured,
            encoder_s=self.encoder_times[1] - self.encoder_times[0],
            first_token_s=timer.times[0] - featured,
        )


def load(spec: ModelSpec, device: str, compute_type: str | None, beams: int) -> HfWhisper:
    """The fetched checkpoint of `spec` on `device`; compute_type is fp16 or fp32."""
    compute_type = compute_type or DEFAULT_COMPUTE_TYPE[device]
    if compute_type not in DTYPES:
        raise ValueError(f"unknown compute type {compute_type!r}; known: {sorted(DTYPES)}")
    if not (spec.source_dir / "config.json").exists():
        raise FileNotFoundError(f"{spec.source_dir} not found; run `make fetch MODEL={spec.name}`")
    return HfWhisper(spec, device, compute_type, beams)
