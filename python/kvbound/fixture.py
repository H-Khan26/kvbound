"""Generates the committed fixtures: random models small enough to commit, plus their goldens.

fixtures/tiny-llama is a 2-layer Llama. It mirrors the primary model's variants: grouped-query
attention, tied embeddings and Llama 3 RoPE scaling.

fixtures/tiny-whisper is a Whisper with 2 encoder and 2 decoder layers, in the layout of a
converted checkpoint (kvbound.models.convert_whisper): a zero bias for every k_proj, no proj_out
tensor, and the decoding rules in generation.json. beam_traces.safetensors holds reference beam
searches over its golden prompts, step by step (see beam_trace).

CI has no GPU and no real weights, so every CPU test (Python and C++) runs against these
fixtures. They have the same file layout as real checkpoints (config.json + model.safetensors
with Hugging Face tensor names), so the same loaders are exercised.

Usage: python -m kvbound.fixture [--beam-traces]  (--beam-traces rewrites the traces alone)
"""

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from kvbound import golden
from kvbound.models import GENERATION_FILE, sha256_file, write_json
from kvbound.paths import FIXTURES
from kvbound.ref import (
    Llama,
    LlamaConfig,
    Whisper,
    WhisperConfig,
    WhisperRules,
    load_llama,
    load_whisper,
)
from kvbound.ref import beam as ref_beam
from kvbound.ref import whisper as ref_whisper
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


def write_random_llama(directory: Path, overrides: dict[str, Any], seed: int = SEED) -> None:
    """A random checkpoint like the fixture's, with config values replaced, and no goldens: for
    tests of shapes the fixture does not have, such as head_dim 64."""
    directory.mkdir(parents=True, exist_ok=True)
    hf_config = HF_CONFIG | overrides
    with (directory / "config.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(hf_config, indent=2) + "\n")
    gen = torch.Generator().manual_seed(seed)
    weights = random_weights(LlamaConfig.from_hf_dict(hf_config), gen)
    save_file(weights, str(directory / "model.safetensors"), metadata={"format": "pt"})


# head_dim 64, the size the tensor-core prefill attention needs and both real models have.
HEAD_DIM_64 = {"hidden_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1}


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


WHISPER_FIXTURE_DIR = FIXTURES / "tiny-whisper"
# The weights and prompts of this seed fit every role of whisper_prompts.
WHISPER_SEED = 796
# Decoder ids fed with each golden prompt: the decoding prompt alone, a short sequence, and one
# that fills every target position.
WHISPER_PROMPT_LENGTHS = (4, 12, 32)
# Smallest logit margin between a greedy step's token and its runner-up on a golden prompt's
# greedy path. An FP16 engine is within a few thousandths of a logit of the reference, so it
# takes the same path.
GREEDY_MARGIN = 0.05
# Audio windows build_whisper draws before giving up on a seed.
MAX_DRAWS = 64

# Special tokens sit at the top of the vocabulary, in Whisper's order.
WHISPER_EOS, WHISPER_START, WHISPER_LANGUAGE, WHISPER_TRANSCRIBE, WHISPER_NO_TIMESTAMPS = (
    120,
    121,
    122,
    123,
    124,
)
# Like openai/whisper-*: the prompt names language and task, a few text tokens and the control
# tokens are never generated, and the first generated token is neither id 5 nor the end.
WHISPER_RULES = WhisperRules(
    prompt=[WHISPER_START, WHISPER_LANGUAGE, WHISPER_TRANSCRIBE, WHISPER_NO_TIMESTAMPS],
    suppress_tokens=[1, 2, 7, WHISPER_START, WHISPER_TRANSCRIBE, 125, 126, 127],
    begin_suppress_tokens=[5, WHISPER_EOS],
    eos_token_id=WHISPER_EOS,
    max_length=32,
)

# head_dim is 64, the size the tensor-core attention kernel needs and every Whisper has. The
# generation fields make Hugging Face `generate` follow WHISPER_RULES.
HF_WHISPER_CONFIG = {
    "architectures": ["WhisperForConditionalGeneration"],
    "model_type": "whisper",
    "vocab_size": 128,
    "num_mel_bins": 16,
    "d_model": 128,
    "encoder_layers": 2,
    "encoder_attention_heads": 2,
    "encoder_ffn_dim": 256,
    "decoder_layers": 2,
    "decoder_attention_heads": 2,
    "decoder_ffn_dim": 256,
    # 128 mel frames in, 64 encoder positions.
    "max_source_positions": 64,
    "max_target_positions": 32,
    "activation_function": "gelu",
    "scale_embedding": False,
    "bos_token_id": WHISPER_EOS,
    "eos_token_id": WHISPER_EOS,
    "pad_token_id": WHISPER_EOS,
    "decoder_start_token_id": WHISPER_START,
    "forced_decoder_ids": [[i, token] for i, token in enumerate(WHISPER_RULES.prompt) if i > 0],
    "suppress_tokens": WHISPER_RULES.suppress_tokens,
    "begin_suppress_tokens": WHISPER_RULES.begin_suppress_tokens,
    "max_length": WHISPER_RULES.max_length,
}

# Token embeddings are small and decoder positions large next to the layers' outputs, so that
# greedy decoding moves through the vocabulary. With O(1) token embeddings the tied output
# projection makes every step repeat its input token.
EMBED_SCALE = 0.3
POSITION_SCALE = 2.0


def random_whisper_weights(cfg: WhisperConfig, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """FP16 weights of a converted Whisper checkpoint, scaled so activations stay O(1)."""
    shapes = {name: p.shape for name, p in Whisper(cfg).named_parameters()}
    weights = {}
    for ours, theirs in ref_whisper.hf_weight_names(cfg).items():
        shape = shapes[ours]
        if ours == "encoder.positions":
            w = ref_whisper.sinusoids(*shape)
        elif ours.endswith("k_proj.bias"):
            w = torch.zeros(shape)
        elif "norm" in ours:
            w = float(ours.endswith("weight")) + 0.1 * torch.randn(shape, generator=gen)
        elif ours.endswith("bias"):
            w = 0.1 * torch.randn(shape, generator=gen)
        elif ours == "decoder.embed.weight":
            w = EMBED_SCALE * torch.randn(shape, generator=gen)
        elif ours == "decoder.positions":
            w = POSITION_SCALE * torch.randn(shape, generator=gen)
        else:
            # Linear [out, in] and conv [out, in, kernel] weights, by fan-in.
            w = torch.randn(shape, generator=gen) / shape[1:].numel() ** 0.5
        weights[theirs] = w.to(torch.float16)
    return weights


def whisper_prompts(model: Whisper, gen: torch.Generator) -> list[golden.AudioPrompt]:
    """Golden prompts whose greedy paths, between them, apply every decoding rule.

    Random audio windows are drawn until one fits each role in turn:
      p0  runs to max_length, along a path the suppress list changes.
      p1  stops at an end-of-sequence token within three quarters of max_length.
      p2  starts with a token the begin-suppress list changes.
    Every path keeps GREEDY_MARGIN at each step. Raises when MAX_DRAWS windows do not fill the
    roles: the weights of another seed will.
    """
    cfg, rules = model.cfg, WHISPER_RULES

    def changes_path(features: torch.Tensor, ids: list[int], without: WhisperRules) -> bool:
        """Whether decoding under the rules `without` one list leaves the path `ids`."""
        return ref_whisper.greedy(model, features, without) != ids

    def fits(role: int, features: torch.Tensor, ids: list[int]) -> bool:
        ended = ids[-1] == rules.eos_token_id
        if role == 0:
            runs_out = len(ids) == rules.max_length and not ended
            return runs_out and changes_path(features, ids, replace(rules, suppress_tokens=[]))
        if role == 1:
            return ended and len(ids) <= 3 * rules.max_length // 4
        return changes_path(features, ids, replace(rules, begin_suppress_tokens=[]))

    prompts: list[golden.AudioPrompt] = []
    for _ in range(MAX_DRAWS):
        features = torch.randn(cfg.n_mels, cfg.n_frames, generator=gen)
        scores: list[torch.Tensor] = []
        greedy_ids = ref_whisper.greedy(model, features, rules, scores)
        role = len(prompts)
        if ref_whisper.top2_gap(scores) < GREEDY_MARGIN or not fits(role, features, greedy_ids):
            continue
        n_random = WHISPER_PROMPT_LENGTHS[role] - len(rules.prompt)
        ids = rules.prompt + torch.randint(cfg.vocab_size, (n_random,), generator=gen).tolist()
        prompts.append(golden.AudioPrompt(f"p{role}", features, ids, greedy_ids))
        if len(prompts) == len(WHISPER_PROMPT_LENGTHS):
            return prompts
    raise RuntimeError(f"no audio window fits prompt p{len(prompts)}; change WHISPER_SEED")


BEAM_TRACE_FILE = "beam_traces.safetensors"


@dataclass(frozen=True)
class BeamCase:
    """One traced beam search: a golden prompt's audio under the fixture's rules, with another
    end-of-sequence token or length penalty where given."""

    prompt: str
    name: str
    num_beams: int
    eos_token_id: int = WHISPER_EOS
    length_penalty: float = 1.0

    def rules(self) -> WhisperRules:
        return replace(WHISPER_RULES, eos_token_id=self.eos_token_id)


# Beam 5 under the fixture's rules runs every prompt to max_length. The other cases make a
# frequent text token the end of the sequence, so that between them hypotheses finish early,
# crowd each other out of the kept set, and end the search before max_length; and they vary the
# beam count and the length penalty.
BEAM_CASES = (
    *(BeamCase(prompt, "beam5", 5) for prompt in ("p0", "p1", "p2")),
    BeamCase("p0", "beam1", 1),
    BeamCase("p1", "beam1", 1),
    BeamCase("p1", "beam2", 2),
    BeamCase("p2", "beam3", 3),
    BeamCase("p1", "beam5.eos38", 5, eos_token_id=38),
    BeamCase("p0", "beam5.eos5", 5, eos_token_id=5),
    BeamCase("p2", "beam5.penalty0", 5, length_penalty=0.0),
    BeamCase("p0", "beam5.eos72.penalty0.6", 5, eos_token_id=72, length_penalty=0.6),
    BeamCase("p2", "beam5.eos91.penalty2", 5, eos_token_id=91, length_penalty=2.0),
)


def beam_trace(result: ref_beam.BeamResult) -> dict[str, torch.Tensor]:
    """A beam search as tensors, for kvb::BeamSearch to replay (tests/cpp/beam_test.cpp).

    Rows of a step-major tensor that a step does not use hold -1 ids and -inf scores. The
    search's BeamConfig is file metadata (see whisper_beam_traces).
      ids                  the best hypothesis as a whole sequence, prompt included
      candidate_ids        [steps, num_beams, n_candidates]: what each live beam offered
      candidate_log_probs  the same shape
      beams                [steps, num_beams, 2]: (parent, token) of the next live beams
      beam_scores          [steps, num_beams]
      finished             [n, 3]: (step, parent, token) of every continuation that ended
      finished_scores      [n]
      hypotheses           [kept, max_new_tokens]: the kept hypotheses' tokens, best first
      hypothesis_scores    [kept]
    The search is done after the last step and not before.
    """
    config = result.config
    n_steps, n_beams, k = len(result.steps), config.num_beams, ref_beam.n_candidates(config)
    candidate_ids = torch.full((n_steps, n_beams, k), -1, dtype=torch.int32)
    candidate_log_probs = torch.full((n_steps, n_beams, k), float("-inf"))
    beams = torch.full((n_steps, n_beams, 2), -1, dtype=torch.int32)
    beam_scores = torch.full((n_steps, n_beams), float("-inf"))
    finished, finished_scores = [], []
    for s, step in enumerate(result.steps):
        for b, candidates in enumerate(step.candidates):
            for j, (token, log_prob) in enumerate(candidates):
                candidate_ids[s, b, j] = token
                candidate_log_probs[s, b, j] = log_prob
        for b, beam in enumerate(step.beams):
            beams[s, b] = torch.tensor([beam.parent, beam.token])
            beam_scores[s, b] = beam.score
        finished += [[s, beam.parent, beam.token] for beam in step.finished]
        finished_scores += [beam.score for beam in step.finished]
    hypotheses = torch.full((len(result.hypotheses), config.max_new_tokens), -1, dtype=torch.int32)
    for i, hypothesis in enumerate(result.hypotheses):
        hypotheses[i, : len(hypothesis.tokens)] = torch.tensor(hypothesis.tokens)
    return {
        "ids": torch.tensor(result.ids, dtype=torch.int32),
        "candidate_ids": candidate_ids,
        "candidate_log_probs": candidate_log_probs,
        "beams": beams,
        "beam_scores": beam_scores,
        "finished": torch.tensor(finished, dtype=torch.int32),
        "finished_scores": torch.tensor(finished_scores, dtype=torch.float32),
        "hypotheses": hypotheses,
        "hypothesis_scores": torch.tensor([h.score for h in result.hypotheses]),
    }


def whisper_beam_traces(
    model: Whisper, features: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The traces of BEAM_CASES, named "<prompt>/<case>/<tensor>"; features by prompt id.

    Returns the tensors and the file metadata: one entry, "configs", a JSON object of every
    case's BeamConfig by "<prompt>/<case>". One entry with sorted keys keeps the file's bytes
    the same from run to run, and JSON numbers read back as the same doubles.
    """
    traces, configs = {}, {}
    for case in BEAM_CASES:
        result = ref_beam.beam_search(
            model, features[case.prompt], case.rules(), case.num_beams, case.length_penalty
        )
        name = f"{case.prompt}/{case.name}"
        for tensor_name, tensor in beam_trace(result).items():
            traces[f"{name}/{tensor_name}"] = tensor
        configs[name] = asdict(result.config)
    return traces, {"configs": json.dumps(configs, sort_keys=True)}


def build_beam_traces(directory: Path = WHISPER_FIXTURE_DIR) -> None:
    """Writes beam_traces.safetensors from the fixture's stored weights and golden features.

    It reads the other files and writes no other, so it runs alone after a change to the beam
    search or BEAM_CASES.
    """
    _, tensors = golden.load(directory)
    features = {case.prompt: tensors[f"{case.prompt}/input_features"] for case in BEAM_CASES}
    traces, metadata = whisper_beam_traces(load_whisper(directory), features)
    save_file(traces, str(directory / BEAM_TRACE_FILE), metadata=metadata)


def build_whisper(directory: Path = WHISPER_FIXTURE_DIR, seed: int = WHISPER_SEED) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    cfg = WhisperConfig.from_hf_dict(HF_WHISPER_CONFIG)
    WHISPER_RULES.check(cfg)

    write_json(directory / "config.json", HF_WHISPER_CONFIG)
    write_json(directory / GENERATION_FILE, WHISPER_RULES.to_dict())
    weights_path = directory / "model.safetensors"
    save_file(random_whisper_weights(cfg, gen), str(weights_path), metadata={"format": "pt"})

    # Goldens come from the file as stored, through the same loader real checkpoints use.
    model = load_whisper(directory)
    meta = {
        "name": "tiny-whisper",
        "weights_sha256": sha256_file(weights_path),
        "weights_precision": "fp16",
        "seed": seed,
    }
    golden.dump_whisper(model, whisper_prompts(model, gen), directory, meta)
    build_beam_traces(directory)


if __name__ == "__main__":
    if sys.argv[1:] == ["--beam-traces"]:
        build_beam_traces()
    else:
        build()
        build_whisper()
        print(FIXTURE_DIR)
    print(WHISPER_FIXTURE_DIR)
