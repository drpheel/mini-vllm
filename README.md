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

## One layer, in order

Each token is a vector of 2048 numbers, called the hidden state. A layer reads that vector and writes an updated one. The same layer runs once per token.

| Step | What it does |
| --- | --- |
| Embedding | Replaces a token id with its row from the embedding table. That row is the 2048-number starting vector. |
| RMSNorm | Rescales a vector so its overall magnitude stays near 1, then multiplies by a learned per-dimension weight. Done at the start of attention, at the start of the MLP, and once more before the final scores. |
| Q, K, V projections | Three matrix multiplies. Q is what this token is looking for. K is what this token offers as a match. V is the information this token will pass along if it is matched. |
| RoPE | Rotates pairs of values inside Q and K by an angle that depends on the token's position. That is how the model tells position 0 from position 10. V is left alone. |
| Attention scores | Dot product of a token's Q with earlier tokens' K, divided by `sqrt(64)`. A high score means "pay attention to that earlier token." |
| Causal mask | Sets scores for future positions to a huge negative number so a token can only see itself and tokens before it. |
| Softmax | Turns one row of scores into weights that add up to 1. |
| Weighted V | Mixes the V vectors of earlier tokens using those weights. That mix is the attention output. |
| Output projection | Matrix multiply that joins the 32 attention heads back into one 2048-vector. |
| Residual add | Adds the layer's input vector back onto its output. The layer learns a change, and the original signal stays available to later layers. |
| MLP (SwiGLU) | A small network applied independently to each token. Two matrix multiplies produce a "gate" and an "up" vector of length 8192. SiLU keeps the gate's positive parts and zeroes down the negative ones, the gate multiplies the up vector, and a third matrix multiply returns to length 2048. Another residual add follows. |
| LM head | After the last layer and a final RMSNorm, one more matrix multiply scores every vocabulary id. The highest scores are the tokens the model currently prefers. The weight matrix is the same embedding table. |

Q is split into 32 heads of 64 values. K and V are split into 8 heads. Four query heads share one key/value head. That sharing is grouped-query attention, and it is why the KV cache is 512 wide instead of 2048.

## Prefill and decode

During prefill, attention for the whole prompt is a pair of matrix multiplies: all queries against all keys, then those weights against all values. The mask and softmax kernels run over that score matrix. At the same time, K and V are copied into the KV cache.

The KV cache is the memory decode reuses. It stores K and V for every layer and every position already seen. Decode does not rerun the prompt. It embeds the single new token, computes that token's Q, K, and V, appends K and V to the cache, and attends over every cached position.

The cache is paged. Memory is a pool of fixed blocks, 16 tokens each. A block table says which physical block holds positions 0–15, which holds 16–31, and so on, separately for each layer. The decode attention kernel follows that table. Paging is what production engines use so many sequences can share one cache pool without reserving a full maximum length up front. This engine has one live sequence and at most 512 positions (32 blocks).

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
