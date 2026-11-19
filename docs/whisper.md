# Whisper on the same engine

## Question

kvbound's engine is built around a decoder-only model. This document asks whether the same engine, with its op
programs, paged KV cache, scheduler, CUDA graphs and quantized kernels, can serve an encoder-decoder speech model, and
how it compares in accuracy and speed with the Hugging Face reference and with faster-whisper. On 1,000 Quran
recitation clips on an RTX 3060 Ti it matches Hugging Face FP16 clip by clip, and by model real-time factor it is 12.1
times faster than Hugging Face and 3.5 times faster than faster-whisper `float16` greedy, 9.0 and 2.7 times at beam 5.

## Model and data

Whisper is a speech model in two parts. An encoder turns a 30-second window of audio, as log-mel features, into 1,500
positions; a decoder writes the transcript one token at a time, attending to its own tokens and to the encoder's
output (cross attention). The model is `whisper-base-ar-quran`, Tarteel's fine-tune of Whisper base for Quran
recitation: 6 encoder and 6 decoder layers of width 512.

The clips are `everyayah-test-1000`: 1,000 clips of the test split of `tarteel-ai/everyayah`, 3.79 hours from 36
reciters, drawn by reciter (`configs/evals/everyayah-test-1000.json`). Each clip is decoded as one window, without
timestamps, one clip at a time. The subset is drawn from the 20,914 test clips of at most 30 s; the other 2,559 of the
23,473 are longer than one window. The recordings carry no documented license, so the clips are evaluated only, never
committed or redistributed. faster-whisper's `int8` runs on this machine's CPU with 4 threads, the setting the Sawt
Sakinah app uses.

**Data.** The records (kind `asr`) come from the RTX 3060 Ti (`rtx3060ti-fd7b40d5`, Windows, unlocked clocks) on
2026-11-18, at two commits whose code differs only in where beam candidates are ranked. kvbound and faster-whisper
`float16` are from `6b3b8cd` (`results/raw/rtx3060ti-fd7b40d5/20261118T1044*` to `20261118T1105*`); Hugging Face,
faster-whisper `int8_float16`, faster-whisper on the CPU and the host-ranking beam runs of the Analysis are from
`4089b55` (`20261118T0753*` to `20261118T0954*`). faster-whisper `float16` ran at both and gives the same word errors
and decoder steps on every clip, model RTF 0.0164 and 0.0163 greedy and 0.0209 at both at beam 5, and mean SM clocks
within 5 MHz.

## Method

**Two programs over one weight table.** The encoder program runs the convolution stem as `Unfold` (each frame with its
two neighbours) and a matrix product, then GELU, the position table, pre-norm layers with LayerNorm, and for each
decoder layer the projection of the cross-attention keys and values into a second cache. The decoder program runs the
embedding and positions, self attention over a paged cache, cross attention over the encoder's keys and values, the
MLP, the tied vocabulary projection, the suppression of tokens the rules forbid, and argmax. Both index one table of
weights, loaded once. The new ops are `LayerNorm`, `AddLayerNorm`, `Gelu`, `Unfold`, `AddPositions`,
`DenseAttention` (the encoder's attention, on the tiled tensor-core prefill kernel without the causal mask) and
`SuppressTokens`. Biases are an epilogue of kvbound's own product kernels and a separate kernel after cuBLAS. Fusion
folds each residual add into the LayerNorm after it.

**`AsrEngine`** is a sibling of `Engine` with two caches. The Self cache, of the decoder's own keys and values, is the
paged cache of the Llama engine ([static-vs-paged-memory.md](static-vs-paged-memory.md)) in 16-token blocks. The
Cross cache holds one block of 1,500 positions per sequence, filled by the encoder step. The scheduler gives an
admitted request an encoder step before its first decoder rows; a preempted sequence loses both caches and is encoded
again when readmitted. Decode-shaped steps replay CUDA graphs from the graph cache `Engine` also uses.

**Decoding rules from the checkpoint.** Fetching writes the prompt, the suppressed tokens, the tokens not allowed
first, the end token and the 448-token limit into `generation.json`, as Hugging Face `generate` applies them to this
checkpoint. Its prompt is the start token alone; the model writes the language and task tokens itself.

**Beam search** is Hugging Face's algorithm (`kvb::BeamSearch`). The live beams are the rows of one decoder step. At a
fork each child takes its parent's list of Self blocks, with reference counts, and a block another beam still holds is
copied before a beam writes into it (copy on write). Every beam points at the clip's one Cross block. Each beam's
candidates are ranked on the GPU as a step of the decoder step's CUDA graph (`kernels::beam_candidates`,
`engine/kernels/reductions.cu`): per row of 51,865 FP32 logits, the maximum, the sum of exponentials in FP64, the
log-softmax rounded to FP32, the suppressed tokens left out, and the best beams + 1 (token, log-probability) pairs,
ties to the lower token. Only those pairs are copied, to pinned host memory, where the host search picks the next
beams. `AsrOptions::beam_topk_device = false` copies the logits rows instead and ranks them on the host
(`AsrEngine::beam_candidates_on_host`).

**Quantization** covers the matrices only (INT8 per channel, INT4 in groups of 128); convolution weights, position
tables, biases and norm parameters stay FP16. Either cache can hold INT8.

**Correctness.** `fixtures/tiny-whisper/` is a 2+2-layer random Whisper with goldens of every encoder and decoder site.
The engine matches each site within the FP16 limits, fused and unfused, at FP16, INT8 and INT4 against the
dequantizing reference. Greedy output equals the reference's token for token under both scheduling policies, with and
without graphs, and through preemptions. Beam search reproduces the reference search's committed traces, with blocks
shared between beams or every block private, and GPU and host ranking give the same search bit for bit. On 20 local
recitation clips the slow tests require greedy output to equal Hugging Face FP16 `generate` token for token, or to part
from it first where the FP32 reference scores the two tokens within FP16 noise of each other (`NEAR_TIE` and
`FP16_GAP_NOISE` in `tests/py/test_asr_greedy.py`); beam 5 must equal Hugging Face's beam 5 or end on hypotheses the
FP32 reference scores within `RESCORE_TOLERANCE` of each other, and host ranking must give the same tokens and scores
as GPU ranking.

## Metrics

- **WER**, the word error rate, is the word-level edit distance from the reference (substitutions, insertions and
  deletions) summed over the clips and divided by the reference words, 10,945 here. Both sides are first normalized as
  the Sawt Sakinah app matches words: diacritics removed and one spelling kept for each letter Arabic writes several
  ways (`kvbound.evals.text`). CER is the same at character level.
- **RTF**, the real-time factor, is processing time over audio time, summed over the clips; 0.0047 means 4.7 ms per
  second of audio. The *model* RTF runs from the log-mel features on the host to the token ids on the host. For
  kvbound that is one `transcribe` call: the features copied to the device, the encoder step and every decoder step,
  each waited for (at beam 5 also the ranking and the host search). For Hugging Face it is the features moved to the
  device and `generate`; for faster-whisper its `encode` and `generate` calls. The *end-to-end* RTF adds the log-mel
  front end and detokenization. kvbound and Hugging Face share a front end (`kvbound.audio`, Hugging Face's feature
  extractor); faster-whisper has its own.
- **Encoder pass** is not alike across stacks. kvbound's covers the features copied to the device and rounded to FP16,
  the convolutions, the encoder layers and the cross-attention keys and values of all 6 decoder layers. Hugging Face's
  is the encoder module alone, features already on the device; faster-whisper's is `encode()`, from the features on
  the host. Both project the cross-attention keys and values in the decoder.
- **First token** and **between tokens** exist for kvbound and Hugging Face only; faster-whisper returns all tokens at
  once. **Decoder, ms per step** is, per clip, the model time minus the encoder pass over the decoder steps (generated
  tokens and the end token); it is the one per-token time every stack has. kvbound and Hugging Face take 82.93 steps
  per clip on average and faster-whisper 79.84, because faster-whisper puts the language, task and no-timestamps
  tokens in the prompt.
- **Clocks.** The GPU clock is not locked; NVML samples it once a second over a run. Each stack is compared at the
  clocks its own run had. Latencies are medians over the 1,000 clips.

## Results

| Stack | WER, greedy / beam 5 | CER, greedy / beam 5 | Clips with a word error | RTF, model, greedy / beam 5 | RTF, end to end, greedy / beam 5 |
| --- | --- | --- | --- | --- | --- |
| kvbound, FP16 | 0.58% / 0.48% | 0.17% / 0.13% | 50 / 44 | 0.0047 / 0.0076 | 0.0051 / 0.0080 |
| kvbound, INT8 weights | 0.55% / 0.46% | 0.16% / 0.12% | 47 / 42 | 0.0040 / 0.0076 | 0.0045 / 0.0080 |
| kvbound, INT4 weights | 1.42% / 1.28% | 0.40% / 0.34% | 102 / 97 | 0.0039 / 0.0074 | 0.0043 / 0.0078 |
| kvbound, FP16 weights, INT8 caches | 0.58% / 0.48% | 0.17% / 0.13% | 50 / 44 | 0.0047 / 0.0076 | 0.0051 / 0.0081 |
| Hugging Face, FP16 | 0.58% / 0.48% | 0.17% / 0.13% | 50 / 44 | 0.0568 / 0.0684 | 0.0572 / 0.0689 |
| faster-whisper GPU, `float16` | 0.72% / 0.60% | 0.29% / 0.25% | 53 / 48 | 0.0163 / 0.0209 | 0.0187 / 0.0235 |
| faster-whisper GPU, `int8_float16` | 0.67% / 0.57% | 0.28% / 0.24% | 50 / 43 | 0.0263 / 0.0308 | 0.0287 / 0.0334 |
| faster-whisper CPU, `int8` | 0.66% / 0.65% | 0.27% / 0.26% | 49 / 47 | 0.0616 / 0.0839 | 0.0647 / 0.0870 |

| Stack | Encoder pass, greedy / beam 5 | First token | Between tokens, ms | Decoder, ms per step | SM clock, mean of samples |
| --- | --- | --- | --- | --- | --- |
| kvbound, FP16 | 5.9 / 5.9 ms | 6.6 / 6.7 ms | 0.70 / 1.18 | 0.70 / 1.18 | 1,975 / 1,978 MHz |
| kvbound, INT8 weights | 5.6 / 5.6 ms | 6.3 / 6.3 ms | 0.59 / 1.19 | 0.60 / 1.18 | 1,950 / 1,964 MHz |
| kvbound, INT4 weights | 5.7 / 5.7 ms | 6.3 / 6.3 ms | 0.57 / 1.15 | 0.57 / 1.15 | 1,976 / 1,978 MHz |
| kvbound, FP16 weights, INT8 caches | 5.9 / 5.9 ms | 6.7 / 6.7 ms | 0.70 / 1.18 | 0.71 / 1.18 | 1,977 / 1,978 MHz |
| Hugging Face, FP16 | 9.2 / 6.1 ms | 20.2 / 17.7 ms | 9.19 / 11.02 | 9.22 / 11.03 | 1,163 / 1,963 MHz |
| faster-whisper GPU, `float16` | 14.4 / 14.5 ms | n/a | n/a | 2.61 / 3.38 | 1,972 / 1,978 MHz |
| faster-whisper GPU, `int8_float16` | 18.0 / 17.1 ms | n/a | n/a | 4.29 / 5.06 | 1,483 / 1,598 MHz |
| faster-whisper CPU, `int8` | 469 / 469 ms | n/a | n/a | 4.71 / 8.32 | n/a |

Every column of the second table is greedy / beam 5.

## Analysis

**kvbound and Hugging Face agree clip by clip.** In FP16, greedy and at beam 5, each of the 1,000 clips has the same
number of word errors and the same number of decoder steps in both. The records hold those counts, not the token ids;
the slow tests compare the ids. FP16 weights with INT8 caches give the same per-clip word errors and steps as FP16, at
the same speed (model time within 1%).

**Speed.** Greedy, by model RTF, kvbound FP16 is 12.1 times faster than Hugging Face, 3.5 times faster than
faster-whisper `float16` and 13.1 times faster than faster-whisper `int8` on the CPU, at 0.70 ms per generated token
and 5.9 ms for the encoder pass. At beam 5 it is 9.0 times faster than Hugging Face, 2.7 times faster than
faster-whisper `float16`, 4.0 times faster than `int8_float16` and 11.0 times faster than `int8` on the CPU. Beams cost
kvbound more against its own greedy run than they cost the others: beam 5's model RTF is 1.63 times greedy's in kvbound
FP16, 1.28 times in faster-whisper `float16` and 1.20 times in Hugging Face. Hugging Face's greedy run has the lower
mean SM clock (1,163 against 1,963 MHz at beam 5) and the longer encoder pass (9.2 against 6.1 ms); the clock samples
are not tied to the encoder steps, so the clock is a likely reading of both, not a measured one, and the greedy ratio
against Hugging Face holds at that run's clock.

**INT8 and INT4 weights.** Greedy, INT8 weights take 14% less model time than FP16 and INT4 weights 17% less. INT8
costs no accuracy: it changes the word errors of 3 clips and has 60 errors in all against FP16's 64 (at beam 5, 2 clips,
50 against 52). INT4 does: it changes 75 clips and more than doubles the errors (155 against 64), mostly substitutions
(137 against 58); at beam 5, 69 clips (140 against 52). At beam 5 the speed advantage disappears: a 5-row token takes
1.18 ms in FP16 and 1.19 and 1.15 ms with INT8 and INT4 weights, so a beam token is 1.7 times a greedy token in FP16
and 2.0 times with quantized weights. With five rows, FP16 projections are wider than `f16_gemv_rows` (4) and go to
cuBLAS, each followed by a separate bias kernel (`run_linear`, `engine/model/linear.cpp`); quantized ones are wider
than `quantized_gemm_rows` (3) and go to `gemm_quantized`, which adds its bias itself
([kernel-optimization.md](kernel-optimization.md)). The step also runs the ranking, and forks copy blocks. No record
breaks the 5-row step into these parts.

**Why faster-whisper differs.** Its per-clip word errors equal Hugging Face's on 994 of the 1,000 clips with `float16`
(greedy and at beam 5), 992 and 991 with `int8_float16`, and 984 and 986 with `int8` on the CPU, so its higher WER comes
from a few clips. Its extra errors are mostly deletions: 15 against 4 greedy (`float16`). It differs in more than
arithmetic. With the language set to Arabic it feeds the prompt `[50258, 50272, 50359, 50363]` (start, Arabic,
transcribe, no timestamps), where Hugging Face and kvbound feed `[50258]` and the model writes the other three. Its
front end computes the log-mel features of the clip before padding them to 30 seconds, where the other two pad the
samples first. The records cannot say how much of the difference comes from the prompt, the front end or the
arithmetic. Its front end also adds 32.10 ms per clip on average to end-to-end time, against 5.60 ms for kvbound FP16
and 5.37 ms for Hugging Face (greedy).

**Ablation: beam candidates ranked on the GPU or on the host.** Ranking on the host copies each live beam's logits row
to the host every step. The hypothesis is that the host ranking takes most of each beam step and the GPU idles while it
runs, and that an idle GPU lowers its clock, which also slows the GPU's own work in the step and the encoder pass.
kvbound beam 5 on the 1,000 clips, host ranking / GPU ranking:

| Weights, caches | Between tokens, ms | RTF, model | Encoder pass, ms | SM clock, mean of samples | GPU utilization, mean of samples |
| --- | --- | --- | --- | --- | --- |
| FP16 | 5.09 / 1.18 (4.31x) | 0.0313 / 0.0076 (4.10x) | 9.97 / 5.89 | 1,076 / 1,978 MHz | 37% / 80% |
| INT8 weights | 5.06 / 1.19 (4.26x) | 0.0308 / 0.0076 (4.05x) | 8.85 / 5.61 | 1,155 / 1,964 MHz | 36% / 80% |
| INT4 weights | 4.93 / 1.15 (4.27x) | 0.0300 / 0.0074 (4.04x) | 8.72 / 5.67 | 1,202 / 1,978 MHz | 35% / 79% |
| FP16 weights, INT8 caches | 5.07 / 1.18 (4.28x) | 0.0312 / 0.0076 (4.08x) | 10.08 / 5.91 | 1,067 / 1,978 MHz | 37% / 80% |

- **The search is the same.** At each precision the two runs have the same word errors and decoder steps on every clip.
- **Nothing else moves.** The greedy kvbound runs, which do not rank, have a model RTF within 0.3% at the two commits,
  and faster-whisper `float16` within 0.3% too.
- **The clock differs.** The host-ranking runs average 1,067 to 1,202 MHz, and the INT4 one is tagged
  `clock_varied`; with GPU ranking they average 1,964 to 1,978 MHz, the level of the greedy runs (1,950 to 1,977). The
  encoder pass, which beam search does not touch, matches the greedy runs' times only with GPU ranking. That idle gaps
  lower the clock is the likely cause, read from a profile not kept with the records; the records show the lower
  clock, not its cause. Part of the 4.3x per token is host work removed and part is the GPU's own work at the higher
  clock, and the records do not separate the two.

**What a WER this low means.** The model was fine-tuned on recitations of this text: the clips are the dataset's test
split, but every split reads the same Quran. A WER near zero shows that the stacks agree, not that the task is hard.

## Limitations

- **Timestamps, word alignment and long-form decoding**, which carries text from one 30-second window into the next,
  are not built. The Sawt Sakinah app transcribes with all three (faster-whisper, CPU, `int8`, beam 5), so it cannot
  use kvbound until they exist. Without long-form decoding, the 2,559 test clips longer than 30 s are left out.
- **The subset is not the full test split.** It leans toward the start of each audio page, and its clips average
  13.64 s against 12.61 s over the test clips of at most 30 s.
- **One clip at a time.** The engine can encode several clips in one encoder step and decode them as rows of one step;
  every record here runs one clip at a time.
- **The 5-row decoder step is not profiled.** A profile would say how much of a beam token is cuBLAS and its bias
  kernels, `gemm_quantized`, the ranking and the block copies.
- **One GPU, unlocked clocks.** All records are from the RTX 3060 Ti; Hugging Face greedy (1,163 MHz) and
  faster-whisper `int8_float16` (1,483 and 1,598 MHz) ran at lower mean SM clocks than the other runs.

## Reproduce

```sh
make fetch MODEL=whisper-base-ar-quran          # converts the checkpoint and writes generation.json
uv run python -m kvbound.quantize whisper-base-ar-quran --scheme int8_pc   # and --scheme int4_g
uv run python -m kvbound.evals.asr subset       # the 1,000 clips into data/datasets/, checked against the manifest
uv run python -m kvbound.evals.asr eval --model whisper-base-ar-quran --backend kvbound --compute-type fp16:int8 --beams 5
make bench-asr MODEL=whisper-base-ar-quran      # Hugging Face and faster-whisper, greedy and beam 5
```
