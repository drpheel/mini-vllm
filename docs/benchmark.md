# Benchmark and kernel profile

Measured on the Orin in this box: compute capability 8.7, 8 SMs, 7.4 GB memory. The checkpoint is Llama 3.2 1B Instruct (`model.safetensors`, 2.30 GiB, BF16). This file has 16 transformer blocks. Prompt file `bench/prompt_128.txt` (128 token ids). Decode is greedy, one sequence, so the three paths can be compared token for token.

Nsight Compute 2024.3.1 is installed, and it is the tool that would report achieved bandwidth and occupancy. On this Jetson it stops immediately with `Insufficient privileges to launch app for profiling` unless it is root, because those counters need debugfs. Nsight Systems 2024.5.4 records every kernel and every CUDA API call without that. That trace is what shows whether time sits in the GEMMs or in launch and sync gaps, which is also the CUDA graph question. The raw tables are `docs/benchmark/eager_stats.txt` and `docs/benchmark/graph_stats.txt`.

## Throughput

`./build/cuda_cublas_json <model> bench/prompt_128.txt --bench --steps 32`

One warmup prefill, then a timed prefill, then 32 greedy decode steps. GPU times are CUDA events around the forward. End-to-end adds the host argmax over the 128256 logits. No profiler was attached.

| Path | Prefill | Decode, GPU | Decode, end to end |
| --- | --- | --- | --- |
| Eager (the engine as it ships) | 164.5 ms, 778 tok/s | 37.53 ms/token, 26.6 tok/s | 37.84 ms/token, 26.4 tok/s |
| Async launches, one sync at the end of the step | | 33.19 ms/token, 30.1 tok/s | 33.39 ms/token, 29.9 tok/s |
| CUDA graph replay of that async step | | 32.47 ms/token, 30.8 tok/s | 32.66 ms/token, 30.6 tok/s |

Host sampling is 0.2–0.3 ms/token. It is not the limit.

The async path and the graph produced the same token ids as eager for all 32 steps. The captured graph has 323 nodes.

## Would a CUDA graph help?

A little against today's eager loop, and almost not at all once that loop stops synchronizing.

The eager decode step calls `cudaDeviceSynchronize` inside every SiLU and the embedding gather, and it uses synchronous `cudaMemcpy` to append K and V. Across a 128-token prefill plus 8 decode steps, Nsight Systems counted 299 `cudaDeviceSynchronize` calls and 713 `cudaMemcpy` calls (`eager_stats.txt`). Those waits are overlapped with GPU work, so the API time is not extra latency on top of the kernels. Removing them, and appending K and V with a kernel whose destination pointer is read at launch time, is the async row: **4.3 ms/token, about 12%**.

The CUDA graph is that same async step, captured once and replayed. RoPE position, sequence length, the token id, and the KV destination pointers live in mapped memory, so replay does not rebuild the graph when the position changes or a new page is allocated. Replay is **1.16×** the eager GPU step and **1.02×** the async step (0.7 ms/token). `cudaGraphLaunch` itself was 0.37 ms of CPU time per replay in the graph trace. The graph is not the lever.

The step is the matrix multiplies. In the eager trace, cuBLAS and CUTLASS BF16 GEMMs are **87%** of GPU kernel time. A decode token reads essentially the whole 2.30 GiB checkpoint. Spread across the 32.5 ms graph step, that is about 76 GB/s of weight traffic. Replaying those loads from a graph does not make them shorter.

## What the kernels cost

Eager trace: one 128-token prefill and 8 decode steps, Nsight Systems 2024.5.4, `--trace=cuda,nvtx,osrt`. Absolute times in that file include profiler overhead (the same prefill was 615 ms under nsys and 165 ms without it). The shares and the per-launch averages are the useful part. Decode-only kernels are the ones whose instance counts are multiples of the 8 steps.

| Work | Share of GPU kernel time | Per launch | Where it runs |
| --- | --- | --- | --- |
| `ampere_bf16_*` and CUTLASS BF16 GEMMs | 87.1% | 15 µs to 19.1 ms | Q, K, V, O, SwiGLU, LM head, attention scores |
| `pagedAttentionKernel` | 5.1% | 228 µs | Decode only. 16 launches per token, 3.6 ms/token |
| `softmaxKernel` | 3.0% | 1.10 ms | Prefill, once per layer |
| `siluKernel` | 1.2% | median 13 µs, max 339 µs | Prefill is the long launch; decode is the short one |
| Prefill RMSNorm | 1.1% | 197 µs | Once per token row in the prompt |
| Prefill RoPE | 0.7% | 120 µs | Q and K |
| Prefill residual | 0.6% | 107 µs | |
| Decode RMSNorm | 0.4% | 9.0 µs | 33 launches per token, about 0.30 ms |
| Causal mask | 0.3% | 111 µs | Prefill |
| Decode residual | 0.3% | 6.6 µs | |
| Decode RoPE | 0.3% | 6.5 µs | |
| Embedding gather | <0.1% | 12–20 µs typical | |

The single 19.1 ms GEMM is one launch, so it is prefill (the vocabulary projection over the whole prompt), not decode. The 2.44 ms CUTLASS GEMM also runs once per layer and only in prefill.

Decode RMSNorm, RoPE, and residual are a few microseconds of GPU work. In the eager launch-and-exec table their API-to-completion time is about 0.5–0.6 ms, because each one is stuck behind a synchronize or a synchronous copy. That gap is what the async path and the graph close. It is a few milliseconds per token in total, not the GEMM time.

Paged attention at 228 µs per layer is the largest custom decode kernel. Sixteen layers is 3.6 ms, about a tenth of the eager step. The other nine tenths are the projections.

## Reproduce

```bash
cmake --preset default && cmake --build --preset default

# timings, no profiler
./build/cuda_cublas_json /mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors \
  bench/prompt_128.txt --bench --steps 32

# kernel and API tables
STEPS=8 tools/profile_engine.sh
```

`--bench` runs eager, async, and the graph. `--bench-eager` and `--bench-graph` are the two profiled shapes. `--no-warmup` skips the extra prefill so the trace is one prefill plus the decode steps. `NSYS` overrides the Nsight Systems binary; the run above used 2024.5.4 extracted from the Jetson package.
