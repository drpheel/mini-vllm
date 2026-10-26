# Benchmark and kernel profile

Measured on the Orin in this box: compute capability 8.7, 8 SMs, 7.4 GB memory. The checkpoint is Llama 3.2 1B Instruct (`model.safetensors`, 2.30 GiB, BF16). This file has 16 transformer blocks. Prompt file `bench/prompt_128.txt` (128 token ids). Decode is greedy, one sequence, so the three paths can be compared token for token.

Nsight Systems 2024.5.4 records every kernel and every CUDA API call. That trace shows whether time sits in the GEMMs or in launch and sync gaps. Nsight Compute 2024.3.1 reads throughput and occupancy. On this Jetson those counters need root, because they come from debugfs. The counter run is one eager decode step, plus the prefill elementwise kernels. Summaries are `docs/benchmark/eager_stats.txt`, `docs/benchmark/graph_stats.txt`, `docs/benchmark/ncu_decode.txt`, and `docs/benchmark/ncu_prefill.txt`.

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

The step is the matrix multiplies. In the eager trace, cuBLAS and CUTLASS BF16 GEMMs are **87%** of GPU kernel time. A decode token reads essentially the whole 2.30 GiB checkpoint. Spread across the 32.5 ms graph step, that is about 76 GB/s of weight traffic. The counters put those GEMMs at 53–79% of L2 peak and about 30% of SM peak, so the time left in them is L2 traffic, which a replay still has to move.

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

## Hardware counters

Nsight Compute 2024.3.1, Speed of Light and Launch Statistics, `--clock-control none`. One eager decode step after the 128-token prefill (`--bench-eager --no-warmup --steps 1`), range `decode_eager`. Prefill elementwise kernels are a second pass over the `prefill` range. DRAM throughput counters return `n/a` on this integrated Orin, so the memory figure is L2 or L1 utilization. The device reports a 1.02 GHz max SM clock and a 2 MB L2. Profiled kernels ran at about 810–890 MHz, because the clocks were left unlocked. The first kernel in each range was still at about 305 MHz. Kernel time below is the counter duration at those clocks. The throughput table at the top of this file is the unprofiled benchmark.

Device time for the 243 decode launches sums to **44.7 ms**. That is the same shape as the 37.5 ms eager step, measured while the profiler was replaying kernels. GEMMs are 39.3 ms of it (88%). Paged attention is 3.8 ms. RMSNorm, residual, SiLU, RoPE, and the embedding gather together are 1.6 ms.

| Decode work | Launches | Duration | L2 | SM | Occupancy | What it is |
| --- | --- | --- | --- | --- | --- | --- |
| `64x64` GEMM, grid `(128,1,1)` | 32 | 522 µs | 67% | 30% | 16.5% of 16.7% | Gate and up. Output rows = 64 × 128 = 8192 |
| `64x64` GEMM, grid `(32,1,3)` | 16 | 547 µs | 64% | 29% | 16.5% of 16.7% | Down. 2048 rows, split-K 3, one per layer |
| `64x64` GEMM, grid `(2004,1,1)` | 1 | 7.83 ms | 79% | 32% | 16.6% of 16.7% | LM head. 64 × 2004 = 128256 |
| `128x64` GEMM, grid `(16,1,2)` | 32 | 144 µs | 67% | 28% | 16.0% of 16.7% | Q and O. 2048 rows, split-K 2 |
| `64x64` GEMM, grid `(8,1,1)` | 32 | 45 µs | 53% | 23% | 8.3% of 16.7% | K and V. 512 rows, half a wave per SM |
| `pagedAttentionKernel` | 16 | 235 µs | 1.9% (L1 8.6%) | 7.5% | 16.7% of 66.7% | 32 blocks of 64 threads, 0.25 waves/SM |
| RMSNorm, residual, SiLU, RoPE, gather | 114 | 11–26 µs | 1–4% | 1–3% | one block | Grid `(1,1,1)` |

Projection names come from the tile and the grid: output rows = tile M × grid.x, and the launch counts match one or two of those GEMMs per layer. Time-weighted across the step, memory throughput is **61%** and SM throughput is **27%**. The big GEMMs are already at their occupancy limit (achieved matches theoretical, 142–146 registers, 8 waves/SM on gate and up). The LM head is the fullest, at 79% of L2 with 125 waves.

Paged attention launches one 64-thread block per query head. Thirty-two blocks on 8 SMs is a quarter of a wave, so the kernel waits on latency with most of the SM idle. The other custom decode kernels are a single block (1024 threads, 256 for K RoPE) and about 1.6 ms together. A graph takes the launch gap off those short kernels. The 39 ms of GEMMs still runs as memory traffic.

A separate metric pass on the Q GEMM (`128x64`, grid `(16,1,2)`) asked for `dram__throughput` directly. It came back `n/a`. L2 on that same launch was 88%, SM 35%. The 67% in the table is the average of all 32 Q and O launches in the full section.

Prefill elementwise kernels, same prompt, 146 launches, 16.7 ms of device time. The prefill GEMMs, including the per-head score and attention matmuls, were left out of this pass.

| Prefill kernel | Launches | Duration | Memory | SM | Occupancy |
| --- | --- | --- | --- | --- | --- |
| `softmaxKernel` | 16 | 394 µs | 39% L1, 6% L2 | 77% | 96% of 100%, 42.7 waves |
| `rmsNormKernel` | 33 | 94 µs | 32% L1, 12% L2 | 31% | 62% |
| `siluKernel` | 16 | 185 µs | 39% L2 | 28% | 63% |
| `residualKernel` | 32 | 59 µs | 30% L2 | 16% | 61% |
| `ropeKernel` (Q, 1024 threads) | 16 | 86 µs | 22% L2 | 55% | 63% |
| `causalMaskKernel` | 16 | 37 µs | 17% L2 | 60% | 67% of 100% |
| `ropeKernel` (K, 256 threads) | 16 | 23 µs | 20% L2 | 53% | 86% |
| `embeddingGatherKernel` | 1 | 81 µs | 19% L2 | 17% | 62%, clock still 305 MHz |

Softmax is the busy custom kernel: 77% of SM peak, with the scores staying in L1. The Nsight Systems 1.10 ms average for that kernel includes the launch gap; the device duration is 394 µs. Sixteen layers is 6.3 ms. The rest of these kernels are a few milliseconds and sit well under both peaks. The 165 ms prefill is still the GEMMs.

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

Counters need root. The push/pop NVTX filter is the range name plus `]`.

```bash
sudo -n /usr/local/cuda/bin/ncu --clock-control none \
  --nvtx --nvtx-include 'decode_eager]' \
  --section SpeedOfLight --section LaunchStats \
  --kernel-name-base demangled --print-summary per-kernel \
  ./build/cuda_cublas_json /mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors \
  bench/prompt_128.txt --bench-eager --no-warmup --steps 1

sudo -n /usr/local/cuda/bin/ncu --clock-control none \
  --nvtx --nvtx-include 'prefill]' \
  --kernel-name 'regex:softmaxKernel|causalMaskKernel|ropeKernel|rmsNormKernel|residualKernel|siluKernel|embeddingGatherKernel' \
  --section SpeedOfLight --section LaunchStats \
  --kernel-name-base demangled --print-summary per-kernel \
  ./build/cuda_cublas_json /mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors \
  bench/prompt_128.txt --bench-eager --no-warmup --steps 1
```
