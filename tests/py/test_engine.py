"""The engine end to end: every capture site against the goldens, and greedy generation against
the reference model token for token."""

import gc

import numpy as np
import pytest
import torch

from kvbound import golden, quantize, tolerance
from kvbound.fixture import FIXTURE_DIR
from kvbound.models import load_spec
from kvbound.ref import load_llama

pytestmark = pytest.mark.gpu


def captured_sites(engine, ids, split):
    """Feeds ids[:split] in one step, then the rest one token per step, capturing every site."""
    parts = [engine.feed(ids[:split], start=0, capture=True)]
    parts += [engine.feed([t], start=i, capture=True) for i, t in enumerate(ids) if i >= split]
    return {site: np.concatenate([part[site] for part in parts]) for site in parts[0]}


def assert_matches_goldens(engine, golden_dir, splits):
    """Per-site rel-RMS, pooled top-1 mismatches and mean KL within the manifest's FP16 limits."""
    manifest, tensors = golden.load(golden_dir)
    limits = manifest["thresholds"]["fp16"]
    for split_of in splits:
        mismatches = 0
        for prompt in manifest["prompts"]:
            ids = tensors[f"{prompt['id']}/input_ids"].tolist()
            sites = captured_sites(engine, ids, split_of(len(ids)))
            for site, value in sites.items():
                expected = tensors[f"{prompt['id']}/{site}"]
                assert value.shape == tuple(expected.shape), site
                error = tolerance.rel_rms(torch.from_numpy(value), expected)
                assert error <= limits["rel_rms_max"][site], (prompt["id"], site)
            logits, expected = torch.from_numpy(sites["logits"]), tensors[f"{prompt['id']}/logits"]
            mismatches += int((logits.argmax(-1) != expected.argmax(-1)).sum())
            assert tolerance.mean_kl(logits, expected) <= limits["mean_kl_max"], prompt["id"]
        assert mismatches <= limits["top1_mismatches_max"]


# Whole prompt as one prefill step; one token per step; a prefill followed by decode steps.
SPLITS = (lambda n: n, lambda n: 1, lambda n: max(1, n // 2))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"plan": "naive"},
        {"kv_block_size": 4},
        {"kv_layout": "token_major"},
        {"kv_layout": "dim_major", "kv_block_size": 8},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "default",
)
def test_fixture_sites_match_goldens(kvb, options):
    engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=64, **options)
    assert_matches_goldens(engine, FIXTURE_DIR, SPLITS)


def test_graph_replay_matches_goldens(kvb):
    """Capturing sites turns graphs off, so the graph path is checked through its logits."""
    manifest, tensors = golden.load(FIXTURE_DIR)
    limit = manifest["thresholds"]["fp16"]["rel_rms_max"]["logits"]
    engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=64, graph=True)
    for prompt in manifest["prompts"]:
        ids = tensors[f"{prompt['id']}/input_ids"].tolist()
        # One-token steps are decode-shaped: the first captures a graph, the rest replay it.
        steps = [engine.feed([t], start=i)["logits"] for i, t in enumerate(ids)]
        logits = torch.from_numpy(np.concatenate(steps))
        assert tolerance.rel_rms(logits, tensors[f"{prompt['id']}/logits"]) <= limit


def test_engine_recovers_after_a_failed_request(kvb):
    engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=64)
    prompt = fixture_prompts()[1]
    expected = engine.generate(prompt, 8)
    with pytest.raises(RuntimeError):
        engine.generate(prompt, 8, top_p=0.0)  # rejected by the sampler
    with pytest.raises(RuntimeError):
        engine.generate([10_000], 8)  # token id beyond the vocabulary
    assert engine.generate(prompt, 8) == expected
    with pytest.raises(RuntimeError):
        engine.serve([prompt, [10_000]], [4, 4])  # fails after the first request was admitted
    assert engine.serve([prompt], [8]) == [expected]
    # feed() and generate() can share an engine.
    logits = engine.feed(prompt)["logits"]
    assert int(logits[-1].argmax()) == expected[0]


def reference_greedy(model, prompt, n_new):
    """Greedy continuation by full-sequence forward passes, and the top-2 logit gap per step."""
    device = model.embed.weight.device
    ids, gaps = list(prompt), []
    for _ in range(n_new):
        logits = model(torch.tensor([ids], device=device))[0, -1].float()
        top = logits.topk(2)
        gaps.append(float(top.values[0] - top.values[1]))
        ids.append(int(top.indices[0]))
    return ids[len(prompt) :], gaps


def fixture_prompts():
    manifest, tensors = golden.load(FIXTURE_DIR)
    return [tensors[f"{p['id']}/input_ids"].tolist() for p in manifest["prompts"]]


def test_fixture_greedy_matches_reference(kvb):
    model = load_llama(FIXTURE_DIR)
    engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=64)
    for prompt in fixture_prompts():
        expected, _ = reference_greedy(model, prompt, 40)
        assert engine.generate(prompt, 40) == expected


def test_static_batches_match_reference(kvb):
    """More requests than slots, of different lengths: each still gets the reference's tokens."""
    model = load_llama(FIXTURE_DIR)
    prompts = [*fixture_prompts(), [9, 8, 7], [200, 3], list(range(40, 60)), [5]]
    lengths = [12, 3, 25, 1, 17, 6, 9]
    expected = [reference_greedy(model, p, n)[0] for p, n in zip(prompts, lengths, strict=True)]
    for options in ({"max_seqs": 3}, {"max_seqs": 4, "max_step_tokens": 7, "kv_block_size": 8}):
        engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=64, **options)
        assert engine.serve(prompts, lengths) == expected
        assert engine.serve(prompts[:2], lengths[:2]) == expected[:2]  # the engine is reusable


def test_continuous_batches_match_reference(kvb):
    """Continuous batching over 4-token blocks gives each request the reference's tokens: in a
    roomy pool, in one tight enough to preempt and recompute sequences, with staggered arrivals,
    and when the engine may keep only one CUDA graph."""
    model = load_llama(FIXTURE_DIR)
    prompts = [*fixture_prompts(), [9, 8, 7], [200, 3], list(range(40, 60)), [5]]
    lengths = [12, 3, 25, 1, 17, 6, 9]
    expected = [reference_greedy(model, p, n)[0] for p, n in zip(prompts, lengths, strict=True)]
    options = {"max_ctx": 64, "max_seqs": 4, "kv_block_size": 4}

    engine = kvb.Engine(str(FIXTURE_DIR), **options)
    assert engine.serve(prompts, lengths, policy="continuous") == expected
    stats = engine.serve_stats()
    assert stats["preemptions"] == 0 and stats["max_running"] == 4
    static_steps = (engine.serve(prompts, lengths), engine.serve_stats()["steps"])[1]
    assert stats["steps"] < static_steps  # no slot idles until its batch ends

    # 16 blocks hold 64 tokens, one full sequence; the requests running together need more.
    tight = kvb.Engine(str(FIXTURE_DIR), kv_blocks=16, max_step_tokens=9, **options)
    assert tight.serve(prompts, lengths, policy="continuous") == expected
    assert tight.serve_stats()["preemptions"] > 0
    arrivals = [0.0, 0.0, 0.01, 0.02, 0.02, 0.05, 0.3]
    assert tight.serve(prompts, lengths, policy="continuous", arrivals_s=arrivals) == expected
    assert engine.serve(prompts, lengths, arrivals_s=arrivals) == expected  # static, as they come

    one_graph = kvb.Engine(str(FIXTURE_DIR), max_graphs=1, **options)
    assert one_graph.serve(prompts, lengths, policy="continuous") == expected
    graphs = one_graph.graph_stats()
    assert graphs["evicted"] > 0 and graphs["captured"] == graphs["evicted"] + 1
    assert graphs["replayed"] > 0


@pytest.fixture(scope="module")
def quantized_fixture(tmp_path_factory):
    """The fixture checkpoint in each quantized scheme. Its rows are 64 and 128 wide, so INT4
    uses groups of 32."""
    dirs = {}
    for scheme in ("int8_pc", "int4_g"):
        dirs[scheme] = tmp_path_factory.mktemp(scheme)
        quantize.convert(FIXTURE_DIR, dirs[scheme], scheme, group_size=32)
    return dirs


@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_quantized_weights_match_the_dequantizing_reference(kvb, quantized_fixture, scheme):
    """The engine on quantized weights against the reference model computing in FP32 on the same
    weights dequantized. Both ways of multiplying are covered: in place, and expanded to FP16 for
    the tensor-core product (quantized_gemm_rows=0 expands on every step)."""
    model_dir = quantized_fixture[scheme]
    model = load_llama(model_dir)
    manifest, tensors = golden.load(FIXTURE_DIR)
    limits = manifest["thresholds"]["fp16"]["rel_rms_max"]
    prompts = fixture_prompts()
    lengths = [12, 30, 20]
    expected_tokens = [
        reference_greedy(model, p, n)[0] for p, n in zip(prompts, lengths, strict=True)
    ]
    for options in ({}, {"quantized_gemm_rows": 0}):
        engine = kvb.Engine(str(model_dir), max_ctx=64, max_seqs=2, kv_block_size=8, **options)
        assert engine.weights() == scheme
        for prompt in manifest["prompts"]:
            ids = tensors[f"{prompt['id']}/input_ids"].tolist()
            expected = golden.capture_sites(model, ids)
            for split in (len(ids), 1):
                for site, value in captured_sites(engine, ids, split).items():
                    error = tolerance.rel_rms(torch.from_numpy(value), expected[site])
                    assert error <= limits[site], (prompt["id"], site, options)
        assert engine.serve(prompts, lengths, policy="continuous") == expected_tokens
    # A quantized checkpoint holds its matrices at the size the format promises.
    fp16_bytes = kvb.Engine(str(FIXTURE_DIR), max_ctx=64).memory()["weights"]
    assert engine.memory()["weights"] < (0.62 if scheme == "int8_pc" else 0.42) * fp16_bytes


def test_int8_kv_cache_is_close_to_fp16_and_the_same_on_every_path(kvb):
    """Quantizing the cache moves the logits by a small error, halves the pool, and gives the
    same tokens whatever the layout, block size, batching or preemption."""
    manifest, tensors = golden.load(FIXTURE_DIR)
    fp16 = kvb.Engine(str(FIXTURE_DIR), max_ctx=64)
    int8 = kvb.Engine(str(FIXTURE_DIR), max_ctx=64, kv_dtype="int8")
    for prompt in manifest["prompts"]:
        ids = tensors[f"{prompt['id']}/input_ids"].tolist()
        exact = torch.from_numpy(fp16.feed(ids)["logits"])
        logits = torch.from_numpy(int8.feed(ids)["logits"])
        assert tolerance.rel_rms(logits, exact) < 0.02
        assert tolerance.top1_agreement(logits, exact) == 1.0
    # 64 one-byte elements and a two-byte scale per vector, against 64 two-byte elements.
    head_dim = manifest["model"]["config"]["hidden_size"] // manifest["model"]["config"]["n_heads"]
    ratio = int8.memory()["kv_pool"] / fp16.memory()["kv_pool"]
    assert ratio == pytest.approx((head_dim + 2) / (2 * head_dim))

    prompts = [*fixture_prompts(), [9, 8, 7], [200, 3], list(range(40, 60)), [5]]
    lengths = [12, 3, 25, 1, 17, 6, 9]
    expected = [int8.generate(p, n) for p, n in zip(prompts, lengths, strict=True)]
    for options in (
        {"max_seqs": 3},
        {"max_seqs": 4, "kv_block_size": 4, "kv_blocks": 16, "max_step_tokens": 9},
        {"max_seqs": 2, "kv_layout": "token_major", "wide_key_loads": False},
        {"max_seqs": 2, "kv_layout": "dim_major", "kv_block_size": 8, "graph": False},
    ):
        engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=64, kv_dtype="int8", **options)
        assert engine.serve(prompts, lengths, policy="continuous") == expected, options


def test_generation_is_the_same_on_every_path(kvb):
    """Graphs, chunked prefill, small KV blocks and the CPU sampler all give the greedy tokens."""
    prompt = fixture_prompts()[-1]
    expected = kvb.Engine(str(FIXTURE_DIR), max_ctx=64, graph=False).generate(prompt, 30)
    for options in (
        {"graph": True},
        {"max_step_tokens": 5},
        {"kv_block_size": 4},
        {"wide_key_loads": False},
    ):
        assert kvb.Engine(str(FIXTURE_DIR), max_ctx=64, **options).generate(prompt, 30) == expected
    engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=64)
    # top_k=1 leaves one candidate, so sampling on the CPU must agree with the device argmax.
    assert engine.generate(prompt, 30, temperature=0.7, top_k=1) == expected
    # The fixture's distribution is sharply peaked; a high temperature flattens it.
    hot = {"temperature": 30.0, "top_k": 50, "top_p": 0.95}
    sampled = engine.generate(prompt, 30, seed=7, **hot)
    assert sampled == engine.generate(prompt, 30, seed=7, **hot)
    assert sampled != engine.generate(prompt, 30, seed=8, **hot)


def test_generate_stops_at_eos_and_rejects_overflow(kvb):
    engine = kvb.Engine(str(FIXTURE_DIR), max_ctx=32)
    prompt = fixture_prompts()[1]
    tokens = engine.generate(prompt, 20)
    stopped = engine.generate(prompt, 20, eos_token_id=tokens[3])
    assert stopped == tokens[: tokens.index(tokens[3]) + 1]
    with pytest.raises(RuntimeError, match="max_ctx"):
        engine.generate(prompt, 32)
    with pytest.raises(RuntimeError, match="at least one prompt token"):
        engine.generate([], 4)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["llama-3.2-1b", "tinyllama-1.1b"])
def test_real_model_matches_goldens_and_reference(kvb, name):
    spec = load_spec(name)
    if not (spec.golden_dir / golden.GOLDEN_FILE).exists():
        pytest.skip(f"{name} has no goldens; run `make fetch golden MODEL={name}`")
    manifest, tensors = golden.load(spec.golden_dir)
    prompts = [tensors[f"{p['id']}/input_ids"].tolist() for p in manifest["prompts"]]
    n_new = 48

    # The FP32 reference and the engine do not fit in 8 GB together, so the reference goes first.
    model = load_llama(spec.checkpoint_dir).to("cuda")
    expected = [reference_greedy(model, prompt, n_new) for prompt in prompts]
    del model
    gc.collect()
    torch.cuda.empty_cache()

    engine = kvb.Engine(str(spec.checkpoint_dir), max_ctx=256)
    assert_matches_goldens(engine, spec.golden_dir, SPLITS[:2])
    for prompt, (tokens, gaps) in zip(prompts, expected, strict=True):
        actual = engine.generate(prompt, n_new)
        if actual != tokens:
            # FP16 arithmetic may flip a choice only where the reference itself was nearly tied.
            first = next(i for i, (a, b) in enumerate(zip(actual, tokens, strict=True)) if a != b)
            pytest.fail(f"diverged at new token {first}, reference top-2 gap {gaps[first]:.4f}")
