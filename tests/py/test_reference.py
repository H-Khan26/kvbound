"""The from-scratch reference model against Hugging Face and against the committed goldens."""

import pytest
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from kvbound import golden, tolerance
from kvbound.fixture import FIXTURE_DIR
from kvbound.models import load_spec
from kvbound.ref import load_llama


def assert_matches_hf(checkpoint_dir, prompts, thresholds):
    ref = load_llama(checkpoint_dir, weights="native")
    hf = LlamaForCausalLM.from_pretrained(
        checkpoint_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    n_layers = ref.cfg.n_layers
    for input_ids in prompts:
        ids = torch.tensor([input_ids])
        capture = {}
        ref(ids, capture)
        with torch.no_grad():
            out = hf(ids, output_hidden_states=True)

        # HF hidden_states: [embed, layer outputs..., final-norm output].
        pairs = [("embed", out.hidden_states[0]), ("final_norm", out.hidden_states[-1])]
        pairs += [(f"layers.{i}.resid_post", out.hidden_states[i + 1]) for i in range(n_layers - 1)]
        pairs.append(("logits", out.logits))
        for site, expected in pairs:
            assert tolerance.rel_rms(capture[site], expected) <= thresholds["rel_rms_max"], site
        assert tolerance.top1_agreement(capture["logits"], out.logits) == 1.0


def test_reference_matches_hf_on_fixture():
    manifest, tensors = golden.load(FIXTURE_DIR)
    prompts = [tensors[f"{p['id']}/input_ids"].tolist() for p in manifest["prompts"]]
    assert_matches_hf(FIXTURE_DIR, prompts, manifest["thresholds"]["fp32"])


def test_committed_goldens_match_reference():
    manifest, tensors = golden.load(FIXTURE_DIR)
    fp32 = manifest["thresholds"]["fp32"]
    model = load_llama(FIXTURE_DIR, weights=manifest["model"]["weights_precision"])
    for prompt in manifest["prompts"]:
        ids = tensors[f"{prompt['id']}/input_ids"].tolist()
        assert len(ids) == prompt["n_tokens"]
        for site, value in golden.capture_sites(model, ids).items():
            expected = tensors[f"{prompt['id']}/{site}"]
            assert value.shape == expected.shape, site
            assert tolerance.rel_rms(value, expected) <= fp32["rel_rms_max"], site
            assert tolerance.cosine(value, expected) >= fp32["cosine_min"], site


@pytest.mark.slow
@pytest.mark.parametrize("name", ["llama-3.2-1b", "tinyllama-1.1b"])
def test_reference_matches_hf_on_real_model(name):
    checkpoint = load_spec(name).checkpoint_dir
    if not (checkpoint / "model.safetensors").exists():
        pytest.skip(f"{name} is not fetched; run `make fetch MODEL={name}`")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompts = [tokenizer(text).input_ids for text in golden.REAL_MODEL_PROMPTS]
    assert_matches_hf(checkpoint, prompts, golden.FP32_THRESHOLDS)
