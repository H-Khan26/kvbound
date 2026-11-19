# From a roofline to tokens per second: an LLM inference engine written by hand

kvbound is a small LLM inference engine in C++ and CUDA, written from scratch to answer one question with
measurements: when a 1B-parameter model generates text on a consumer GPU, where do the time and the memory go? This
post follows the project from the first hardware measurement to the comparison with production stacks. Every number
comes from a record in the repository, measured on an RTX 3060 Ti (8 GB) with TinyLlama 1.1B unless it says
otherwise.

## Start with the ceiling

Generating one token at batch size 1 multiplies a single vector by every weight matrix in the model. Each weight is
read once and used once, so the step cannot finish before the weights have crossed the memory bus. That makes
decode a memory-bandwidth problem, and it gives a ceiling that needs no engine to compute:

```text
tokens per second <= memory bandwidth / bytes of weights read per token
```

A microbenchmark that sweeps the working-set size measured 415 GB/s on this card beyond its cache. TinyLlama's
weights are 2.07 GB in FP16, so a decode step takes at least 5.0 ms and the ceiling is 200 tokens per second.
Hugging Face Transformers reached 17 to 33 on the same card. A long prompt is the opposite case: all its tokens go
through each matrix together, the arithmetic dominates, and the limit is the tensor cores' peak rate. Plotting both
limits against arithmetic per byte gives the roofline, and every later result in the project is read against it.

## Correct before fast

Before any CUDA, the model was written from scratch in PyTorch and matched to Hugging Face. Its activations at
every stage of the forward pass were saved as golden tensors, with the error each stage may show in FP16. The C++
engine has to stay inside those limits at every stage and generate the same tokens as the reference, and that
contract was checked again after every optimization.

The contract settled one design decision early. Matrix products that accumulate their sums in FP16 run at about
58 TFLOPS on this card, and with FP32 accumulation at 33. FP16 accumulation failed the contract, so the engine pays
for FP32.

The first working engine used cuBLAS for the matrix products and decoded at 144 tokens per second, 72% of the
ceiling. A profile showed 90% of the GPU time in the products, as the model predicted, and one surprise on the CPU
side: a decode step was 357 kernel launches, and launching them one by one cost 1.5 ms. Capturing the step in a
CUDA graph, which replays all the launches with one call, cut that to 0.13 ms.

## Two ways to hold the KV cache

A transformer keeps a key and a value for every token it has processed, in every layer: the KV cache. For
TinyLlama that is 22,528 bytes per token. How an engine reserves that memory decides how many requests it can
serve at once, and the project built two designs on the same engine to compare them.

**The static design** is how runtimes for phone NPUs work. The whole cache is allocated up front as 8 slots of
1,024 tokens (176 MiB), each admitted request owns a slot, and a batch runs until its longest request ends. It is
simple, and every address is known before the first token. Its cost shows when requests differ in length. On 48
requests with long-tailed lengths, only 15% of the reserved cache held tokens, the slots were busy 38% of the time,
and the engine served 451 tokens per second.

**The paged design** is how GPU serving systems work. The same 176 MiB is cut into 16-token blocks. A request
takes blocks as it grows, a block table per sequence maps its positions to blocks, and requests join and leave
between any two steps (continuous batching). On the same long-tailed requests the same memory served 1,469
tokens per second, with 89% of the reserved cache holding tokens and up to 48 sequences in one step. When requests
arrive over time at 8 per second, the median wait for the first token fell from 1.9 s to 13 ms.

Paging has costs, and the project measured them too:

- **Each token arrives more slowly.** Steps carry more sequences, so on equal-length requests the median time per
  output token rose from 7.5 to 13.0 ms while total throughput rose from 955 to 1,776 tokens per second.
- **The block table is on the hot path.** Attention looks up a block for every cached position it reads. Doing
  that with a division made attention 1.37 to 1.84 times slower than an unpaged cache. With a power-of-two block
  size it is a shift and a mask, and the cost drops to 1.07 to 1.12 times.
- **Admission is a guess.** The engine cannot know how long a response will be, so a full pool sometimes has to
  evict the newest sequence and compute it again later.

## Fewer bits, fewer bytes, more tokens

If decode speed is bytes over bandwidth, smaller weights are faster weights. The engine stores weights as INT8
with a scale per output channel, or as INT4 with a scale per group of 128, and multiplies them without expanding
them in memory:

| TinyLlama 1.1B | Weights on device | Decode, batch 1 | WikiText-2 perplexity |
| --- | --- | --- | --- |
| FP16 | 2,098 MiB | 177 tokens/s | 8.87 |
| INT8 | 1,050 MiB | 314 tokens/s | 8.87 |
| INT4, round to nearest | 541 MiB | 507 tokens/s | 9.96 |

INT8 is free in accuracy. INT4 is not: rounding each weight to the nearest of 15 levels raises perplexity by 12%
here and by 34% on Llama 3.2 1B. Two published methods, implemented on the engine's own format, recover most of
that without changing a kernel. AWQ rescales the input channels of each matrix so that channels with large
activations are rounded more finely. GPTQ rounds one column at a time and pushes each column's rounding error into
the columns not yet rounded. Together they bring INT4 to 9.20 on TinyLlama and 12.13 on Llama 3.2, where FP16
scores 8.87 and 10.99.

A third method did not help. Rotating the model's hidden state by a random orthogonal matrix before quantizing,
which spreads outlier values across channels, made plain rounding worse on both models. With GPTQ it changed nothing
on TinyLlama (9.30 against 9.31) and little on Llama 3.2 (12.27 against 12.37), where it also needs a second copy of
the vocabulary matrix, a fifth more weights to store. The negative result is recorded with the others.

The KV cache can be quantized as well. An INT8 cache does not make a single request faster, because attention is a
few percent of a decode step. It halves the bytes per cached token, so the same pool runs 58 sequences at once
instead of 30.

## Closing the gap to the roofline

With correct FP16 and quantized paths in place, the last engine phase went back to the roofline and asked of each
gap what was left on the table. Each change had a hypothesis, a switch to turn it off, and a before and after
measured back to back:

- **A fused decode step.** Stacking the query, key and value matrices into one product, folding residual adds into
  the normalization after them, and folding the rotary embedding into the cache write cut a layer from 16
  operations to 9.
- **A hand-written FP16 matrix-vector product** in place of cuBLAS for decode. It streams a decode step's matrices
  at 394 GB/s of the 415 the card can deliver.
- **Quantized weights on tensor cores.** Steps with many rows used to expand each INT4 matrix to FP16 first, which
  made quantized models slower than FP16 under load. A kernel that expands weights in registers and feeds the
  tensor cores directly reversed that: 96 concurrent requests now run at 2,290 tokens per second on INT4 weights
  against 1,804 on FP16.
- **Prefill attention on tensor cores**, through the block table. A 1,024-token prompt went from 202 ms to 73 ms.

FP16 decode ended at 177 tokens per second, 88% of the ceiling, and Llama 3.2 1B at 92% of its own. INT4 stops at
66%: the weights now cross the bus in 1.3 ms, but the rest of a step, about 0.7 ms of normalization, attention and
small kernels, did not shrink with them. An RTX 4070 repeats the pattern with its own numbers: 1.14 times the
bandwidth gives 1.19 to 1.33 times the decode speed, and 1.81 times the tensor-core rate gives a 1,024-token prompt
in 39 ms.

## Against production stacks

The last measurement put the engine next to software people deploy. llama.cpp ran on the same card with the same
token ids, the same requests and the same bytes of KV memory.

**One request.** kvbound decodes faster: 180 tokens per second against 160 in FP16, and 510 against 381 at 4 bits
(both engines measured in one later session on a newer driver, so kvbound's figures sit a little above the table's).
Profiles of both engines show that llama.cpp's decode step is about 0.7 ms longer at either precision, and what the
0.7 ms is made of. It multiplies the query, key and value matrices as three products where kvbound stacks them into
one. Its 4-bit format spends 4.5 bits per weight where kvbound's spends 4.125. It quantizes the activations to 8 bits
before every quantized product, a second kernel each time. It waits on the GPU twenty times in a step where kvbound
waits once. None of these is a deep difference, and each is what the roofline says to look for: bytes read per
token, and time not spent reading them.

**A long prompt.** Here llama.cpp is ahead. It evaluates 1,024 tokens in 59 to 68 ms where kvbound's first token
takes 70 to 73. kvbound sums its matrix products in FP32, the price of the accuracy contract from the second
section; llama.cpp sums in FP16, which the tensor cores do almost twice as fast.

**Serving.** While requests arrive at up to 8 per second, both engines keep up and deliver the same throughput.
With 96 requests at once kvbound serves 1,819 tokens per second against 1,340, and 1,917 against 1,026 on Llama
3.2. Faster kernels are not the reason: llama.cpp's engine alone steps 21 sequences in 10 ms, and kvbound steps 30
in 12. One reason is admission. llama.cpp never preempts, so its slots must be few enough that every
request can finish, 21 in this memory; the paged design admits by free blocks and runs 30. The other is the server
around the engine, which samples each sequence's token on the CPU and adds 4 to 8 ms to every step.

**Accuracy per bit.** llama.cpp's plain 4-bit format, with one scale per 32 weights, scores 9.29 on TinyLlama where
kvbound's plain rounding with one scale per 128 scores 9.96. AWQ and GPTQ bring kvbound to 9.20 with fewer bits,
but they calibrate on text from the same corpus as the test, and llama.cpp's files use no calibration, so the
comparison is not settled. One step up in size, llama.cpp's most used format takes 5.2 bits per weight on Llama 3.2
and scores 11.44 against kvbound's best 12.13. kvbound has no format in between to answer with.

The second comparison was a different kind of system: Qualcomm's export of Llama 3.2 1B for the Snapdragon NPU,
profiled on two phones through Qualcomm AI Hub. It is the static design from earlier in this post, in production
form: a context of 4,096 positions compiled into the graph, one sequence, and every step attending over the full
context under a mask. Its per-graph latencies work out to 58 to 60 tokens per second. On the NPU the softmax and the
attention products are the costliest operators; on the GPU engine, attention is a few percent of a step.

<!-- STACKS -->

## The same engine, a different model

To check that the engine was more than one model's forward pass, the project ran Whisper on it: a speech model with
an encoder, a decoder, and a second cache for the encoder's output. Beam search reuses the paged cache, with beams
sharing blocks until one of them writes. On 1,000 clips of Quran recitation the engine makes the same word errors
as Hugging Face Transformers on every clip and runs 12 times faster, and 3.5 times faster than faster-whisper, a
production speech stack.

## What is not done

The project stops with open items, and the repository lists them where they belong. The larger ones:

- Every number here is from Windows, where the GPU clock cannot be locked. Records whose clock moved during the run
  are tagged, and none of this is publication-grade.
- INT4 decode stops at 66% of its ceiling. The 0.7 ms of a step that is not weights is the next thing to shrink.
- The 4-bit accuracy comparison needs llama.cpp quantized with calibration data and kvbound's checkpoints scored on
  text from another corpus.
- Requests with a common prompt prefix do not share cache blocks, although the reference counts that would allow it
  exist for beam search.
- The speech path has no timestamps and no audio longer than one 30-second window.
<!-- OPEN -->

The repository has the code, every record behind these numbers, a design document on the two memory designs, the
comparison with production stacks, and the log of each optimization with its profiler evidence.
