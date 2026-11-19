"""A ctypes binding of the pinned release's llama.dll: load a GGUF, feed token ids, read logits.

Just enough of include/llama.h at the pinned tag to score perplexity the way kvbound does: a
window of token ids is fed from position 0 with the cache cleared, as one batch with every
position's logits requested, and the logits come back as an array [n, vocab]. The structs below
mirror the tag's llama_model_params and llama_context_params field by field; a different tag
needs them checked again, and Llama.__init__ checks the defaults it reads back.

The ggml backends are separate DLLs in the release folder, loaded with
ggml_backend_load_all_from_path, and the CUDA runtime comes from the release's cudart folder.
"""

import ctypes
import os
from ctypes import POINTER, byref, c_bool, c_char_p, c_float, c_int, c_int32, c_size_t, c_void_p
from pathlib import Path
from typing import Any

import numpy as np

from kvbound.baselines import llamacpp

# enum llama_flash_attn_type, enum llama_process_type and enum ggml_type values at the tag.
FLASH_ATTN = {"auto": -1, "off": 0, "on": 1}
PROCESS_DECODE = 1
GGML_TYPES = {"f16": 1, "q4_0": 2, "q8_0": 8}


class ModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32),
        ("split_mode", c_int),
        ("load_mode", c_int),
        ("lazy_mode", c_int),
        ("main_gpu", c_int32),
        ("tensor_split", c_void_p),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
        ("load_mtp", c_bool),
    ]


class ContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("ctx_type", c_int),
        ("rope_scaling_type", c_int),
        ("pooling_type", c_int),
        ("attention_type", c_int),
        ("flash_attn_type", c_int),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int),
        ("type_v", c_int),
        ("moe_cache_size", c_size_t),
        ("moe_cache_cpu_share", c_float),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("samplers", c_void_p),
        ("n_samplers", c_size_t),
        ("ctx_other", c_void_p),
    ]


# name: (restype, argtypes) of every llama.dll function used.
FUNCTIONS: dict[str, tuple[Any, list[Any]]] = {
    "llama_backend_init": (None, []),
    "llama_backend_free": (None, []),
    "llama_model_default_params": (ModelParams, []),
    "llama_context_default_params": (ContextParams, []),
    "llama_model_load_from_file": (c_void_p, [c_char_p, ModelParams]),
    "llama_model_free": (None, [c_void_p]),
    "llama_init_from_model": (c_void_p, [c_void_p, ContextParams]),
    "llama_free": (None, [c_void_p]),
    "llama_model_get_vocab": (c_void_p, [c_void_p]),
    "llama_vocab_n_tokens": (c_int32, [c_void_p]),
    "llama_n_ctx": (ctypes.c_uint32, [c_void_p]),
    "llama_get_memory": (c_void_p, [c_void_p]),
    "llama_memory_clear": (None, [c_void_p, c_bool]),
    "llama_batch_ext_init": (c_void_p, [c_void_p]),
    "llama_batch_ext_free": (None, [c_void_p]),
    "llama_batch_ext_clear": (None, [c_void_p]),
    "llama_batch_ext_add_token": (c_int32, [c_void_p, c_int32, c_int32]),
    "llama_batch_ext_set_pos": (c_bool, [c_void_p, c_int32, POINTER(c_int32)]),
    "llama_batch_ext_set_output_logits": (c_bool, [c_void_p, c_int32, c_bool]),
    "llama_process": (c_int32, [c_void_p, c_int, c_void_p]),
    "llama_get_logits": (POINTER(c_float), [c_void_p]),
    "llama_tokenize": (
        c_int32, [c_void_p, c_char_p, c_int32, POINTER(c_int32), c_int32, c_bool, c_bool],
    ),
}  # fmt: skip


def load_library() -> ctypes.CDLL:
    """llama.dll of the pinned release, with its backends registered."""
    for folder in (llamacpp.BIN, llamacpp.CUDART):
        if not folder.exists():
            raise SystemExit(f"{folder} not found; run `make fetch-llamacpp`")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(folder))
    ggml = ctypes.CDLL(str(llamacpp.BIN / "ggml.dll"))
    ggml.ggml_backend_load_all_from_path.argtypes = [c_char_p]
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path(str(llamacpp.BIN).encode())
    lib = ctypes.CDLL(str(llamacpp.BIN / "llama.dll"))
    for name, (restype, argtypes) in FUNCTIONS.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    lib.llama_backend_init()
    return lib


def check_defaults(model: ModelParams, context: ContextParams) -> None:
    """Raises when the defaults read back do not look like the tag's: a sign that the structs
    above no longer match llama.h."""
    seen = (context.n_ctx, context.n_batch, context.n_ubatch, context.n_seq_max)
    if seen != (512, 2048, 512, 1) or context.type_k != GGML_TYPES["f16"]:
        raise RuntimeError(f"llama_context_params does not match llama.h at {llamacpp.TAG}")
    if not (context.offload_kqv and context.op_offload) or model.vocab_only:
        raise RuntimeError(f"llama.h at {llamacpp.TAG} and the ctypes structs disagree")


class Llama:
    """One model and one context of n_ctx positions; every position of a fed batch outputs
    logits."""

    def __init__(
        self, gguf: Path, n_ctx: int, *, flash_attn: str = "on", kv: str = "f16", threads: int = 0
    ) -> None:
        self.lib = lib = load_library()
        model_params = lib.llama_model_default_params()
        context_params = lib.llama_context_default_params()
        check_defaults(model_params, context_params)
        model_params.n_gpu_layers = -1  # every layer on the GPU
        self.model = lib.llama_model_load_from_file(str(gguf).encode(), model_params)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {gguf}")
        context_params.n_ctx = n_ctx
        context_params.n_batch = n_ctx
        context_params.n_ubatch = n_ctx
        context_params.n_seq_max = 1
        context_params.flash_attn_type = FLASH_ATTN[flash_attn]
        context_params.type_k = context_params.type_v = GGML_TYPES[kv]
        if threads:
            context_params.n_threads = context_params.n_threads_batch = threads
        self.settings = {
            "n_ctx": n_ctx,
            "n_batch": n_ctx,
            "n_ubatch": n_ctx,
            "n_gpu_layers": "all",
            "flash_attn": flash_attn,
            "type_k": kv,
            "type_v": kv,
            "n_threads": context_params.n_threads,
        }
        self.ctx = lib.llama_init_from_model(self.model, context_params)
        if not self.ctx:
            raise RuntimeError("llama.cpp could not create a context")
        self.vocab = lib.llama_model_get_vocab(self.model)
        self.n_vocab = lib.llama_vocab_n_tokens(self.vocab)
        self.batch = lib.llama_batch_ext_init(self.ctx)

    def logits(self, ids: list[int]) -> np.ndarray:
        """Logits [len(ids), vocab] of ids fed from position 0 after clearing the cache. The
        array is a view of llama.cpp's buffer, valid until the next call."""
        lib = self.lib
        lib.llama_memory_clear(lib.llama_get_memory(self.ctx), True)
        lib.llama_batch_ext_clear(self.batch)
        for pos, token in enumerate(ids):
            idx = lib.llama_batch_ext_add_token(self.batch, 0, token)
            if idx < 0:
                raise RuntimeError(f"llama_batch_ext_add_token returned {idx} at position {pos}")
            lib.llama_batch_ext_set_pos(self.batch, idx, byref(c_int32(pos)))
            lib.llama_batch_ext_set_output_logits(self.batch, idx, True)
        status = lib.llama_process(self.ctx, PROCESS_DECODE, self.batch)
        if status != 0:
            raise RuntimeError(f"llama_process returned {status}")
        return np.ctypeslib.as_array(lib.llama_get_logits(self.ctx), shape=(len(ids), self.n_vocab))

    def tokenize(self, text: str, add_special: bool = True) -> list[int]:
        """llama.cpp's own tokenization of text, BOS added when the vocabulary asks for it."""
        data = text.encode("utf-8")
        capacity = len(data) + 2
        out = (c_int32 * capacity)()
        n = self.lib.llama_tokenize(self.vocab, data, len(data), out, capacity, add_special, False)
        if n < 0:
            raise RuntimeError(f"llama_tokenize needs {-n} tokens")
        return list(out[:n])

    def close(self) -> None:
        lib = self.lib
        if self.batch:
            lib.llama_batch_ext_free(self.batch)
        if self.ctx:
            lib.llama_free(self.ctx)
        if self.model:
            lib.llama_model_free(self.model)
        self.batch = self.ctx = self.model = None
