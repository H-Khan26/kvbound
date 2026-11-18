"""Speech recognition with kvbound's speech engine: features -> token ids -> text.

Whisper is the model: the engine encodes each clip's 30 s window of log-mel features, then
decodes under the checkpoint's decoding rules (generation.json, kvbound.ref.whisper.WhisperRules):
the rules' prompt first, suppressed tokens never chosen, the begin-suppressed ones not chosen
first, until the end-of-sequence token or max_length tokens. It decodes greedily, or with beams by
Hugging Face's beam search (kvbound.ref.beam), one clip at a time, the beams of a step being the
rows of one decoder step. Text is the generated ids through the checkpoint's tokenizer.json with
special tokens skipped.

The module is also a backend of kvbound.evals.asr (`--backend kvbound`): load() returns a
Transcriber. Its compute type names the weights, as a variant written by kvbound.quantize, and
optionally the dtype of both KV caches after a colon: fp16, int8_pc, int4_g, fp16:int8 and so on.

Times, in seconds on the host:
  model_s        features on the host to token ids on the host: one transcribe() call of the
                 engine, which copies the features to the device, encodes, and runs every decoder
                 step, each one waited for. The same for every clip of the call.
  encoder_s      the engine's encoder step, waited for: the features copied to the device and
                 rounded to FP16, the convolutions, the encoder layers and the projection of every
                 decoder layer's cross-attention keys and values.
  first_token_s  features on the host to the first generated token on the host, by the engine's
                 clock, which starts once the features are in its transcribe() call.
  token_s        when each generated token was on the host, by the same clock; the differences
                 are the per-token decode times. With beams, when the decoder step that chose the
                 best hypothesis's token at that position had its candidates on the host.

With beams, model_s is one beam-search call of the engine per clip.

Usage: python -m kvbound.evals.asr eval --model whisper-base-ar-quran --backend kvbound
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from kvbound import audio, native
from kvbound.evals.asr import Transcript
from kvbound.models import ModelSpec

TOKENIZER_FILE = "tokenizer.json"
KV_DTYPES = ("fp16", "int8")
# Hugging Face's default, which the speech baselines run with.
LENGTH_PENALTY = 1.0


@dataclass
class Transcription:
    """One clip through the engine."""

    text: str
    # Ids fed before the first generated one: the rules' prompt.
    prompt: list[int]
    # Generated ids after the prompt, without the end-of-sequence token.
    tokens: list[int]
    # Whether decoding stopped on the end-of-sequence token rather than at max_length.
    ended: bool
    model_s: float
    encoder_s: float
    first_token_s: float
    # One per generated id, the end-of-sequence token included.
    token_s: list[float]
    # The best hypothesis's beam-search score; None when decoded greedily.
    score: float | None = None


class Recognizer:
    """A Whisper checkpoint on the GPU with its tokenizer.

    options are AsrEngine options (max_seqs, max_encode_batch, graph, fuse, kv_dtype, ...); the
    engine reports them resolved in engine.config().
    """

    def __init__(self, model_dir: Path, tokenizer_file: Path | None = None, **options: Any) -> None:
        model_dir = Path(model_dir)
        self.engine = native.load().AsrEngine(str(model_dir), **options)
        self.rules = self.engine.rules()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file or model_dir / TOKENIZER_FILE))

    def transcribe(self, features: np.ndarray, policy: str = "static") -> list[Transcription]:
        """Features [clips, n_mels, n_frames] or [n_mels, n_frames] as float32, all in one run
        of the engine under the scheduling policy ("static" or "continuous")."""
        started = time.perf_counter()
        features = np.ascontiguousarray(features, dtype=np.float32)
        results = self.engine.transcribe(features, policy=policy)
        model_s = time.perf_counter() - started
        eos = self.rules["eos_token_id"]
        transcriptions = []
        for result in results:
            ids = list(result["ids"])
            ended = bool(ids) and ids[-1] == eos
            tokens = ids[:-1] if ended else ids
            token_s = list(result["token_s"])
            transcriptions.append(
                Transcription(
                    text=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                    prompt=list(self.rules["prompt"]),
                    tokens=tokens,
                    ended=ended,
                    model_s=model_s,
                    encoder_s=result["encode_s"],
                    first_token_s=token_s[0],
                    token_s=token_s,
                )
            )
        return transcriptions

    def transcribe_beam(
        self, features: np.ndarray, num_beams: int, length_penalty: float = LENGTH_PENALTY
    ) -> list[Transcription]:
        """Features [clips, n_mels, n_frames] or [n_mels, n_frames] as float32, one clip after
        the other by beam search; the engine needs max_seqs >= num_beams."""
        features = np.ascontiguousarray(features, dtype=np.float32)
        eos = self.rules["eos_token_id"]
        transcriptions = []
        for clip in features.reshape(-1, *features.shape[-2:]):
            started = time.perf_counter()
            result = self.engine.transcribe_beam(
                clip, num_beams=num_beams, length_penalty=length_penalty
            )[0]
            model_s = time.perf_counter() - started
            ids = list(result["ids"])
            ended = bool(ids) and ids[-1] == eos
            tokens = ids[:-1] if ended else ids
            # Step i chose position i of every beam, the best hypothesis's included.
            token_s = list(result["step_s"])[: len(ids)]
            transcriptions.append(
                Transcription(
                    text=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                    prompt=list(self.rules["prompt"]),
                    tokens=tokens,
                    ended=ended,
                    model_s=model_s,
                    encoder_s=result["encode_s"],
                    first_token_s=token_s[0],
                    token_s=token_s,
                    score=result["score"],
                )
            )
        return transcriptions


class KvboundWhisper:
    """The engine behind kvbound.evals.asr.Transcriber: one clip per call."""

    def __init__(self, spec: ModelSpec, weights: str, kv: str, beams: int = 1) -> None:
        model_dir = spec.variant_dir(weights)
        if not (model_dir / "model.safetensors").exists():
            fix = f"make fetch MODEL={spec.name}"
            if weights != "fp16":
                fix = f"python -m kvbound.quantize {spec.name} --scheme {weights}"
            raise FileNotFoundError(f"{model_dir} not found; run `{fix}`")
        self.n_mels = int(spec.arch["n_mels"])
        self.beams = beams
        # The beams of a clip take one block-table row each.
        self.recognizer = Recognizer(
            model_dir,
            spec.checkpoint_dir / TOKENIZER_FILE,
            kv_dtype=kv,
            cross_kv_dtype=kv,
            max_seqs=beams,
        )
        config = self.recognizer.engine.config()
        self.engine = {"name": "kvbound", "preset": "cuda"}
        self.params = {
            "backend": "kvbound",
            "device": "cuda",
            "compute_type": weights if kv == "fp16" else f"{weights}:{kv}",
            "weights": config["weights"],
            "kv": kv,
            "beams": beams,
            "engine": config,
            "model_s_covers": (
                "features on the host to token ids on the host: the features copied to the "
                "device, the encoder step and every decoder step, each waited for"
                + ("" if beams == 1 else ", with the beams' candidates picked on the host")
            ),
            "encoder_ms_covers": (
                "the engine's encoder step, waited for: features copied to the device and "
                "rounded to FP16, convolutions, encoder layers and the cross-attention keys and "
                "values of every decoder layer"
            ),
            "first_token_ms_covers": (
                "features on the host to the first generated token on the host, timed inside "
                "the engine's transcribe call"
                + ("" if beams == 1 else ": the first decoder step's beam candidates")
            ),
        }
        if beams > 1:
            self.params["length_penalty"] = LENGTH_PENALTY
            self.params["beam_share_blocks"] = config["beam_share_blocks"]

    def __call__(self, samples: np.ndarray) -> Transcript:
        started = time.perf_counter()
        features = audio.features(samples, self.n_mels).numpy()
        if self.beams == 1:
            result = self.recognizer.transcribe(features)[0]
        else:
            result = self.recognizer.transcribe_beam(features, self.beams)[0]
        return Transcript(
            text=result.text,
            prompt=result.prompt,
            tokens=result.tokens,
            e2e_s=time.perf_counter() - started,
            model_s=result.model_s,
            encoder_s=result.encoder_s,
            first_token_s=result.first_token_s,
        )


def load(spec: ModelSpec, device: str, compute_type: str | None, beams: int) -> KvboundWhisper:
    """The checkpoint of `spec` in the engine; compute_type is "<weights>" or "<weights>:<kv>"."""
    if device != "cuda":
        raise ValueError("kvbound runs on cuda only")
    if beams < 1:
        raise ValueError(f"beams must be at least 1, not {beams}")
    if spec.family != "whisper":
        raise ValueError(f"{spec.name} is a {spec.family}, not a speech model")
    weights, _, kv = (compute_type or "fp16").partition(":")
    kv = kv or "fp16"
    if kv not in KV_DTYPES:
        raise ValueError(f"unknown KV dtype {kv!r}; known: {KV_DTYPES}")
    return KvboundWhisper(spec, weights, kv, beams)
