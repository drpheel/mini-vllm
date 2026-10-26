# CUDA Llama 3.2 1B engine

This program reads a prompt and writes the model's reply, one token at a time, on an NVIDIA GPU. The model is Llama 3.2 1B Instruct. The math is written out in this repo, so you can follow a transformer forward pass from the embedding lookup to the choice of the next token.

A token is one integer id from the model's vocabulary (about 128,000 ids). The program never sees raw text. `tools/tokenizer.py` turns text into those ids, and can turn generated ids back into text.

The deep version of this, with each kernel's launch shape and the KV-cache layout, is in [docs/engine.md](docs/engine.md).

## What a run does

1. Load the weights from a safetensors file into one GPU allocation.
2. **Prefill.** Run every prompt token through all 32 layers. Save the attention memory for that prompt.
3. **Decode.** Pick a next token, run only that token through the 32 layers, append it to the memory, and repeat.
4. Print the generated token ids.

Prefill exists because the prompt is known up front, so every prompt token can be processed together. Decode exists because each new token depends on the tokens chosen before it, so it has to run after the previous choice.

## Where each operation runs

A token's hidden state is 2048 BF16 numbers. Three regions touch it. The embedding runs once before any layer. Each of the 32 layers then runs the same list of steps on its own weights. The vocabulary scores run once after the last layer.

The layer body is the same code 32 times. What makes layer 7 different from layer 8 is the weight tensors `model.layers.7.*` and that layer's own pages in the KV cache. RoPE, the mask, softmax, SiLU, and the residual adds have no learned weights, so they are identical at every layer. Prefill and decode call those same steps. They differ in how many tokens are in each multiply, and in the attention step in the middle of the layer.

### Once, before layer 0

The embedding gather replaces each token id with its row of `model.embed_tokens.weight`. Prefill gathers every prompt token. Decode gathers the single new token. This table is shared by the whole model. It is not inside the layer loop. The LM head reads it again at the end.

### Inside every layer

Prefill's loop is `for (layer ...)` in `llama_prefill::prefill` in `src/prefill.cu`. Decode's loop is the matching `for (layer ...)` in `src/main.cu`. Both walk layers 0 through 31, and both index that layer's weights.

| Step | What is unique to this layer | Prefill, all prompt tokens | Decode, the one new token |
| --- | --- | --- | --- |
| RMSNorm | `input_layernorm.weight` | Every prompt row | The new token's row |
| Q, K, V | `q_proj`, `k_proj`, `v_proj` | One GEMM per matrix, over the whole prompt | The same GEMM with a token count of 1 |
| RoPE | Nothing learned. The angle depends on the token's position. | Positions `0 .. P-1` | The absolute index of the new token |
| Store K and V | This layer's pages in the block table | Copies the whole prompt into fresh pages | Copies one token into the page for its position |
| Attention | This layer's stored K and V | Score GEMM, causal mask, softmax, then scores times V, all over the prompt | `pagedAttentionKernel`, told this layer's index, reads every page for this layer |
| Output projection | `o_proj.weight` | GEMM over the prompt | GEMM of the single attention output |
| Residual add | Nothing learned | Adds the hidden state saved at the start of this layer | Adds the new token's hidden state |
| RMSNorm | `post_attention_layernorm.weight` | Every prompt row | The new token's row |
| MLP | `gate_proj`, `up_proj`, `down_proj` | Gate and up GEMMs, SiLU, down GEMM, over the prompt | The same four steps for one token |
| Residual add | Nothing learned | Adds the MLP output onto the prompt hidden states | Adds it onto the new token |

Q is split into 32 heads of 64 values. K and V are split into 8 heads. Four query heads share one key/value head. That sharing is grouped-query attention, and it is why each layer's K and V are 512 wide instead of 2048.

Attention is the step that is actually a different operation in the two passes. Prefill still has every key in one dense buffer, so it builds a prompt-by-prompt score matrix, masks future positions, and mixes V with a GEMM. Decode's keys are already scattered through this layer's pages, including the token just written, so one kernel walks those pages. The mask is unnecessary there: the pages only contain positions up to the current token.

The KV cache holds K and V separately for every layer. A page is 16 tokens. The block table maps "layer 7, positions 16–31" to a physical page. Prefill fills a layer's pages while that layer runs. Decode appends to the same pages on later steps and reads them back inside `pagedAttentionKernel`.

### Once, after layer 31

A final RMSNorm uses `model.norm.weight`, one vector for the whole model. The LM head then multiplies by the embedding table and writes a score for every vocabulary id. Prefill keeps a row of scores per prompt token and the sampler reads the last row. Decode writes one row. Sampling happens on the CPU after that, outside the layer loop.

## Words that show up everywhere

| Word | What it means here |
| --- | --- |
| safetensors | The checkpoint file. A JSON header lists each tensor's name, shape, and byte offset. The rest of the file is raw weights. |
| BF16 | The 16-bit float format of the weights and activations. Matrix multiplies accumulate in 32-bit float, then store BF16 again. |
| cuBLAS | NVIDIA's matrix-multiply library. Every projection and the attention products go through `cublasGemmEx`. |
| CUDA kernel | A function that runs on the GPU, written in this repo. Embeddings, RMSNorm, RoPE, the mask, softmax, SiLU, residual adds, and decode attention are kernels. |
| GEMM | A general matrix multiply. `Y = X * W^T` is the shape of every projection. |
| Logits | The vocabulary scores at the end of the forward pass, one number per token id. |
| Temperature | Divides the logits before softmax. `0.7` makes the sampler willing to pick strong alternatives. `0` would always pick the top score. |
| Top-p | Nucleus sampling. After softmax, keep the smallest set of tokens whose probabilities add up to `0.9`, then sample inside that set. |
| Best of N | From the same prompt cache, generate 4 candidate replies and keep the one with the best average log-probability, with a small preference for a medium length. |

## Where to read the code

| File | What it owns |
| --- | --- |
| `src/main.cu` | Loads the checkpoint, allocates buffers, runs prefill, then the decode loop: append K and V, call cuBLAS for the projections, sample the next id, score the best-of-N candidates. |
| `src/prefill.cu` | The full-prompt forward pass, plus the kernels decode reuses (embedding gather and SiLU). |
| `src/decode.cu` | The one-token kernels: RMSNorm, RoPE at an explicit position, paged attention, residual add. |
| `src/prompt_tokens.cuh` | Reads a text file of token ids and places them in GPU-visible memory. |
| `tools/tokenizer.py` | Converts between text and token ids with the Llama tokenizer. |
| `docs/engine.md` | Kernel launches, buffer reuse, GEMM layout, and the page format. |

## Build and run

```bash
cmake --preset default
cmake --build --preset default
./build/cuda_cublas_json /path/to/model.safetensors /path/to/prompt_tokens.txt
```

`prompt_tokens.txt` is a list of token ids separated by spaces, commas, or newlines. To make one from text:

```bash
python tools/tokenizer.py "The capital of France is" --chat --model meta-llama/Llama-3.2-1B-Instruct
```

`--chat` wraps the text in the Llama 3 Instruct header so the model answers as the assistant. Copy the printed ids into the prompt file. After a run, decode the printed ids with:

```bash
python tools/tokenizer.py --decode --ids 791 6864 315 --model meta-llama/Llama-3.2-1B-Instruct
```

With no arguments, the binary loads `/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors` and waits for a prompt path as the second argument.

Without CMake presets:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build
```

On Jetson, set the GPU architecture. `87` matches many Orin boards. `72` matches many Xavier boards.

```bash
cmake --preset default -DCMAKE_CUDA_ARCHITECTURES=87
```

Logits, the block table, and the token ids live in mapped host memory, which is the right fit for Jetson's shared CPU/GPU memory. A discrete GPU that returns a separate device address for that mapping will fail at startup.

## Benchmark

Greedy timings, an Nsight Systems trace, and Nsight Compute counters are in [docs/benchmark.md](docs/benchmark.md). On this Orin, a 128-token prefill is 165 ms and eager decode is 37.5 ms/token. A CUDA graph of the decode step matches those tokens and is about 1.16× the eager step. The GEMMs are most of the time, and they already run at about 53–79% of L2 peak.

```bash
./build/cuda_cublas_json /path/to/model.safetensors bench/prompt_128.txt --bench --steps 32
```
