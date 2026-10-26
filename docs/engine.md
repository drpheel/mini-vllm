# How the engine runs Llama 3.2 1B

This engine loads a Llama 3.2 1B Instruct checkpoint (`model.safetensors`, BF16) and generates token IDs on a Jetson-class GPU. Matrix multiplies go through cuBLAS. Everything else — embeddings, RMSNorm, RoPE, the attention mask, softmax, SwiGLU, the residual adds, and decode-time attention — is a small CUDA kernel in `src/prefill.cu` or `src/decode.cu`. The host loop in `src/main.cu` owns weight loading, the paged KV cache, sampling, and the best-of-N decode loop.

The model shape is compiled in:

| Constant | Value | Meaning |
| --- | --- | --- |
| `HIDDEN_SIZE` | 2048 | Residual width |
| `NUM_Q_HEADS` | 32 | Query heads, `2048 / 64` |
| `NUM_KV_HEADS` | 8 | Key and value heads |
| `HEAD_DIM` | 64 | Per-head width |
| `KV_DIM` | 512 | Packed KV width, `8 * 64` |
| `GQA_Q_TO_K_RATIO` | 4 | Query heads that share one KV head |
| `MLP_INTERMEDIATE_SIZE` | 8192 | SwiGLU hidden width |
| `VOCAB_SIZE` | 128256 | Read from `model.embed_tokens.weight` at runtime |
| `N_LAYERS` | 32 | Transformer blocks |
| `BLOCK_SIZE` | 16 | Tokens per KV page |
| `MAX_BLOCKS_PER_SEQ` | 32 | Pages per sequence per layer |
| `MAX_SEQ_LEN` | 512 | `16 * 32`, prompt plus generated tokens |
| `MAX_SEQUENCES` | 1 | One live sequence slot |

RoPE uses Llama 3's base of `500000`. RMSNorm uses `eps = 1e-5`. The LM head is tied to the token embedding table. `lm_head.weight` is loaded if the file contains it, and generation never reads it.

## Glossary

These are the words the rest of this document uses. The first table is the literal expansion of each abbreviation. The tables after it are the meaning in this engine, then the reason that piece is in the forward pass.

| Abbreviation | The letters stand for |
| --- | --- |
| LM | Language model. The LM head is the language-model head: the last layer, which turns the hidden state into scores over the vocabulary. |
| MLP | Multi-layer perceptron. |
| RMS | Root mean square. |
| RMSNorm | Root-mean-square normalization. |
| SiLU | Sigmoid linear unit. |
| GLU | Gated linear unit. |
| SwiGLU | Swish-gated linear unit. Swish is the same curve as SiLU, `x * sigmoid(x)`. |
| Q, K, V | Query, key, value. |
| KV | Key and value. |
| GQA | Grouped-query attention. |
| RoPE | Rotary position embedding. |
| GEMM | General matrix-matrix multiply. |
| BF16 | Brain float, 16-bit. Brain float is a float format named for the research group that published it. |
| FP32 | Floating point, 32-bit. |
| CUDA | Compute Unified Device Architecture. NVIDIA's GPU programming model. |
| cuBLAS | CUDA Basic Linear Algebra Subprograms. The GPU version of the standard BLAS matrix library. |
| BLAS | Basic Linear Algebra Subprograms. |
| GPU | Graphics processing unit. |
| JSON | JavaScript Object Notation. The text format of the safetensors header. |

### The vectors

| Term | What it means | Why it is needed |
| --- | --- | --- |
| Token | An integer id in the vocabulary, about 128,000 possible ids. The program reads and writes ids. Text conversion happens outside, in `tools/tokenizer.py`. | The embedding table is indexed by id. Every later step is arithmetic on the vector that id selects. |
| Embedding | The 2048-number row stored for one token id. | That row is the token's starting hidden state. The same table is reused at the end to score the vocabulary. |
| Hidden state | The 2048-number vector carried through the 32 layers for one token. | It is the token's running representation. Each sublayer reads it and writes an updated version. |
| Activation | Any tensor computed while the model runs: hidden states, Q, scores, gate, logits. Weights are the tensors loaded from the checkpoint. | Activations are the workspace for one forward pass. They are discarded or overwritten. Weights stay for the whole run. |
| Residual | The add of a sublayer's output back onto the hidden state that entered that sublayer. "Residual width" means the 2048-wide stream those adds travel on. | Attention and the MLP learn an update. Adding that update back keeps the representation from the earlier layers available to the later ones. |
| Logits | The final score vector, one number per vocabulary id. | Sampling reads these scores and turns them into the next token id. |

### One layer

| Term | What it means | Why it is needed |
| --- | --- | --- |
| Layer | One of the 32 stacked blocks. Each block is a norm, an attention sublayer, a residual add, a norm, an MLP, and a residual add. | One block mixes tokens and then mixes features. Stacking 32 of them is the capacity Llama 3.2 1B was trained with. |
| RMSNorm | Root-mean-square normalization. For a vector `x`, divide every element by `sqrt(mean(x^2) + 1e-5)`, then multiply by a learned per-element weight. RMS is that square-root scale. | Applied before attention, before the MLP, and once before the logits. It keeps the size of the hidden state from drifting as it passes through 32 layers. |
| Projection | A learned linear map, `Y = X * W^T`, implemented as one GEMM. The named projections are Q, K, V, O, gate, up, and down. | This is how a layer turns the hidden state into queries, keys, values, and the MLP's wide intermediate. The learned `W` is a weight matrix from the checkpoint. |
| MLP | Multi-layer perceptron. Here it is the feed-forward network applied to each token on its own after attention: gate, up, SiLU, down. | Attention mixes information across tokens. The MLP mixes information across the 2048 features of a single token. |
| SwiGLU | The MLP formula Llama uses: `down(silu(gate(x)) * up(x))`. Gate and up each expand 2048 values to 8192. Down returns to 2048. | That is the MLP this checkpoint was trained with. The gate chooses which parts of the up vector survive. |
| SiLU | `x * sigmoid(x)`. Positive values pass through. Negative values shrink toward zero. The kernel writes `silu(gate) * up` in place. | It is the nonlinearity in the MLP. The gate and up projections are linear. SiLU is what keeps them from collapsing into a single linear map. |

### Attention

| Term | What it means | Why it is needed |
| --- | --- | --- |
| Q, K, V | Query, key, and value. Q is what this token is looking for. K is what this token offers as a match. V is the content mixed in when the match is strong. | Attention is a lookup. The score is a Q·K dot product. The output is a weighted sum of V. |
| Head | A slice of 64 values with its own query. This model has 32 query heads and 8 key/value heads. | Separate heads let one token match earlier tokens for several different reasons in the same layer. |
| GQA | Grouped-query attention. Query heads come in groups of 4 that share one K head and one V head. | Llama 3.2 1B was trained this way. The shared heads make the stored K and V 512 wide per token instead of 2048. |
| RoPE | Rotary position embedding. Pairs of values inside Q and K are rotated by an angle that grows with the token's position. The base angle is `500000`. | The Q, K, and V projections have no position input. The rotation makes the Q·K score depend on how far apart the two tokens are. |
| Attention score | `dot(Q, K) / sqrt(64)` for one query position against one key position. | It is the match strength before it becomes a weight. A larger score pulls more of that key's V into the output. |
| Causal mask | Sets the score to `-inf` when the key sits at a later position than the query. | The model was trained to predict the next token from tokens up to the current one. The mask keeps a prompt token from using tokens that come after it. |
| Softmax | `exp(score - max) / sum(exp)` across the keys for one query. | It turns match scores into weights that add up to 1, so the mix of V is a weighted average. The `-inf` mask positions contribute about 0. |
| KV cache | The stored K and V for every layer and every position already seen. | Decode attends to the prompt and to earlier generated tokens by reading this memory. Those positions are not projected again. |
| Page | A fixed block of 16 tokens of K followed by 16 tokens of V. A block table maps a position range in the sequence to a physical page. | The sequence grows one token at a time. Pages let the cache be allocated in chunks of 16 instead of as one dense tensor of the maximum length. |

### Running a forward pass

| Term | What it means | Why it is needed |
| --- | --- | --- |
| Prefill | One forward pass over every prompt token together. It writes K and V for those tokens into the cache. | The prompt is known before any new token is chosen, so its tokens can be scored as a batch and the cache can be filled before generation starts. |
| Decode | One forward pass for a single new token. It appends that token's K and V and attends over the whole cache. | The next token is chosen from the previous step's logits, so its forward pass has to wait for that choice. |
| GEMM | A general matrix-matrix multiply. In this repo each one is a `cublasGemmEx` call. | Every projection, the prefill attention products, and the vocabulary scores are matrix multiplies. cuBLAS is the library that runs them on the GPU. |
| Kernel | A CUDA function compiled into this binary and launched on the GPU. | Gather, RMSNorm, RoPE, the mask, softmax, SiLU, the residual add, and decode attention are reductions or elementwise steps. A GEMM does not express them. |
| BF16 | Brain float16, a 16-bit float with the same exponent range as FP32 and a shorter mantissa. Weights and activations are BF16. Products accumulate in FP32. | The checkpoint is stored in BF16. Keeping activations in that format cuts memory in half relative to FP32. The FP32 accumulation keeps the sums accurate. |
| Safetensors | The checkpoint file. A JSON header names each tensor and gives its byte offset. The rest of the file is the raw weight bytes. | It is how the named Llama weights are found and copied into one GPU allocation. |
| LM head | LM stands for language model. This is the language-model head: the last GEMM, final normalized hidden state times the embedding table, producing logits. "Head" means the output layer of the network. "Tied" means this multiply reuses the embedding weights. A separate `lm_head.weight` in the file is ignored. | It turns the 2048-vector into one score per vocabulary id so a token can be chosen. |

### Choosing the next token

| Term | What it means | Why it is needed |
| --- | --- | --- |
| Temperature | A divisor applied to the logits before softmax. This engine uses `0.7`. A temperature of `0` skips the softmax and takes the largest logit. | It sets how willing the sampler is to pick a strong runner-up along with the top score. |
| Top-p | Nucleus sampling. After softmax, keep the smallest set of tokens whose probabilities add up to `0.9`, drop the rest, and sample inside what remains. | It removes the long tail of tokens the model currently assigns almost no probability. |
| Log-probability | The natural log of the probability the sampler assigned to the token it picked. | Best-of-N ranks whole replies by the average of these values, so a reply the model itself considers likely scores higher. |
| Best of N | Generate 4 full replies from the same prefill cache and keep the one with the best average log-probability, with a small bonus for a medium length. | Extra decode work buys a stronger sample from the same prompt. |

## Where each step lives

| Step | Owner |
| --- | --- |
| Safetensors load, buffer allocation, best-of-N loop, decode KV append, decode GEMMs, sampling | `src/main.cu` |
| Prefill forward, and the gather / RoPE / mask / softmax / SiLU / residual kernels | `src/prefill.cu` |
| Decode RMSNorm, decode RoPE, paged attention, decode residual | `src/decode.cu` |
| Prompt token file and mapped token buffer | `src/prompt_tokens.cuh` |

Decode calls `llama_prefill::embedding_gather` and `llama_prefill::silu` directly. The RMSNorm, RoPE, and residual kernels are duplicated. The math matches. The decode versions take a position or a short token count instead of a whole prompt.

## Two passes, one cache

Prefill consumes the whole prompt in one forward pass. Every token attends to earlier tokens in the prompt. At each layer, the K and V projections are copied into a paged cache. Prefill attention itself does not read that cache. It multiplies the full Q, K, and V activations for the prompt. The cache exists so decode does not have to replay the prompt.

Decode then runs once per new token. It embeds that one token, projects Q, K, and V, rotates Q and K, appends K and V to the cache, and runs paged attention over every cached token, including the one just written. There is no separate causal mask in decode: the cache only contains positions `0 .. seq_len-1`.

```text
prompt tokens
    |
    v
embedding gather
    |
    v
for each of 32 layers:
    RMSNorm -> Q,K,V -> RoPE(Q,K) -> write K,V into pages
    attention over the prompt (prefill) or over the pages (decode)
    O proj -> residual add
    RMSNorm -> gate, up -> SiLU -> down -> residual add
    |
    v
final RMSNorm -> tied embedding GEMM -> logits
    |
    v
host sampler picks the next token id
```

After prefill, `current_prompt_len[slot]` is the prompt length `P`. Pages hold positions `0 .. P-1`. The first sampled token is not in the cache yet. The first decode step embeds it at sequence index `P`, writes its K and V at that position, and attends over `P+1` tokens.

## Weight loading

`read_safetensors_metadata` reads the 8-byte little-endian header length, parses the JSON header, and checks that every tensor is BF16 and that `shape * dtype size` matches `data_offsets`. `copy_payload_to_gpu` then copies the raw payload into one `cudaMalloc` block in 64 MiB chunks. `build_llama_weight_views` points each tensor at `base + data_offset`. No transpose happens on the way in.

Per layer, the views are the Hugging Face names:

- `input_layernorm.weight`, `post_attention_layernorm.weight`
- `self_attn.{q,k,v,o}_proj.weight`
- `mlp.{gate,up,down}_proj.weight`

PyTorch stores each projection as row-major `[out_features, in_features]`. Q and O are `[2048, 2048]`. K and V are `[512, 2048]`. Gate and up are `[8192, 2048]`. Down is `[2048, 8192]`. The embedding table is `[vocab, 2048]`.

## Why every GEMM looks transposed

cuBLAS addresses matrices as column-major. The activations and weights in this engine are row-major. Those two layouts are transposes of each other in the same bytes, and the GEMM flags are chosen so the stored result is the row-major product we want.

For a projection `Y = X * W^T` with `X` row-major `[tokens, in]` and `W` row-major `[out, in]`:

```text
cublasGemmEx(OP_T, OP_N,
             m = out, n = tokens, k = in,
             A = W, lda = in,
             B = X, ldb = in,
             C = Y, ldc = out)
```

cuBLAS computes `C_col = W * X^T`, which is `(X * W^T)^T`. Written column-major with leading dimension `out`, those bytes are row-major `Y [tokens, out]`. Accumulation is FP32 (`CUBLAS_COMPUTE_32F`); the inputs and the store are BF16. Every Q, K, V, O, gate, up, down, and LM-head GEMM in both passes uses this pattern. Attention scores and `scores * V` use the same idea with a head stride. The call shapes for those two are in the kernel section below.

## What each GEMM is for

A GEMM here is one `cublasGemmEx` call. Inputs and outputs are BF16. The product accumulates in FP32. `alpha` is 1 and `beta` is 0 for every call except the prefill attention scores, which scale by `1/sqrt(64)`. `beta = 0` means the call overwrites its output buffer.

Prefill multiplies every prompt token in one call, so the token dimension is `P`. Decode and the benchmark multiply the single new token, so that dimension is 1. The weight matrix and the math are the same in both passes. Per layer, in order:

| GEMM | Math | Result shape | What it is doing |
| --- | --- | --- | --- |
| Q projection | `Q = RMSNorm(h) * W_q^T` | `[tokens, 2048]` | Turns the normalized residual into queries. `W_q` is `[2048, 2048]`. Each group of 64 columns is one query head. RoPE runs on this buffer next. |
| K projection | `K = RMSNorm(h) * W_k^T` | `[tokens, 512]` | Turns the same normalized residual into keys. `W_k` is `[512, 2048]`, eight heads of 64. RoPE runs on K, then the rows are copied into the KV cache. |
| V projection | `V = RMSNorm(h) * W_v^T` | `[tokens, 512]` | Turns that residual into values. `W_v` matches `W_k`. V is not rotated. It is copied into the cache beside K. |
| Attention scores | `scores_i = Q_head_i * K_head_(i/4)^T / 8` | `[P, P]` per query head | Prefill only. One call per query head, 32 calls per layer. Measures how much each prompt token matches each earlier token for that head. Four query heads share one key head. The mask kernel and the softmax kernel run on this matrix before it is used. Decode does not launch these calls. `pagedAttentionKernel` computes the same dots from the paged cache. |
| Scores times V | `attn_head_i = scores_i * V_head_(i/4)` | `[P, 64]`, packed into `[P, 2048]` | Prefill only. One call per query head. Mixes the value vectors with the softmax weights, which is the attention output for that head. The 32 head slices are written back into `q_proj` at columns `i*64`. Decode folds this mix into `pagedAttentionKernel`. |
| O projection | `O = attn * W_o^T` | `[tokens, 2048]` | Joins the 32 head slices back into one residual-sized vector. `W_o` is `[2048, 2048]`. The residual kernel then adds O onto the residual stream saved before this sublayer. |
| MLP gate | `gate = RMSNorm(h) * W_gate^T` | `[tokens, 8192]` | First linear step of the MLP, from the post-attention normalized residual. `W_gate` is `[8192, 2048]`. `siluKernel` overwrites this buffer with `silu(gate) * up`. |
| MLP up | `up = RMSNorm(h) * W_up^T` | `[tokens, 8192]` | The content vector that the gate multiplies. `W_up` matches `W_gate`. It is read by `siluKernel` and then left alone. |
| MLP down | `down = (silu(gate) * up) * W_down^T` | `[tokens, 2048]` | Projects the gated intermediate back to the residual width. `W_down` is `[2048, 8192]`. The residual kernel adds it onto the post-attention stream. |
| LM head | `logits = RMSNorm_final(h) * E^T` | prefill `[P, vocab]`, decode `[1, vocab]` | After all 32 layers. Scores every vocabulary id. `E` is `model.embed_tokens.weight`, shape `[vocab, 2048]`, the same table the embedding gather reads. Prefill writes a row per prompt token and the sampler reads the last row. Decode writes one row into the front of `embed_proj`. |

That is 9 projection GEMMs per layer, plus 64 attention GEMMs per layer in prefill (32 score calls and 32 value mixes), plus one LM head after the last layer. Decode launches the 9 projections and the LM head, and no attention GEMMs. The benchmark in `src/bench.cu` launches that same decode set through one helper. Its `m`, `n`, and `k` match the table with `tokens = 1`.

## Activation buffers

`main` allocates the working set before either pass. Lengths are in elements.

| Buffer | Shape | Role |
| --- | --- | --- |
| `input_embeddings` | `[512, 2048]` | Prompt embeddings. During prefill, also the saved attention residual. |
| `hidden_state` | `[512, 2048]` | Residual stream. Decode writes only the current token's row. |
| `rms_norms` | `[512, 2048]` | RMSNorm output. Decode also parks the O projection here. |
| `q_proj` | `[512, 2048]` | Q, then attention output, then the MLP down projection. |
| `k_proj_batched_buffer` | `[512, 512]` | K for the whole prompt, or the single decode token at row 0. |
| `v_proj_batched_buffer` | `[512, 512]` | Same for V. |
| `mlp_gate`, `mlp_up` | `[512, 8192]` | SwiGLU intermediates. SiLU overwrites `mlp_gate`. |
| `prefill_attn_scores` | `[32, 32, P, P]` | One `[P, P]` score matrix per query head per layer. Sized to the real prompt. |
| `embed_proj` | `[P, vocab]` mapped | Logits. Decode rewrites row 0 only. |

`P` is the prompt length. The other activation buffers are sized to `MAX_PROMPT_LEN` (512) so decode can address a row by sequence index up to that limit.

Several buffers are intentionally overwritten inside a layer:

- Prefill copies `hidden_state` into `input_embeddings` before attention, then the O projection overwrites `hidden_state`, then the residual kernel adds `input_embeddings` back.
- `scores * V` is stored on top of `q_proj`. Q is dead after the score GEMMs.
- The MLP down projection is also stored in `q_proj`, then added into `hidden_state`.
- Decode computes Q at `q_proj + position * 2048`, copies that row to `q_proj[0]` because paged attention indexes `active_slot * 2048`, and writes the attention output back to `q_proj[0]`.

## Paged KV cache

The cache is a pool of fixed pages, not a `[layers, seq, kv]` tensor.

```text
physical block b
  bytes [0, v_offset):              K for 16 tokens, each token 512 BF16
  bytes [v_offset, block_bytes):    V for the same 16 tokens
```

`v_offset` is `16 * 512 * 2` bytes. `block_bytes` is twice that (32 KiB). A token inside the K region is `token_in_block * 512` BF16 values, and head `h` occupies the slice `[h * 64, (h + 1) * 64)`. V uses the same layout after `v_offset`.

The pool has `num_layers * MAX_BLOCKS_PER_SEQ` blocks (1024 for this model). One sequence that reaches 512 tokens uses every block: 32 pages per layer, 32 layers, and pages are not shared across layers.

The block table is an `int` array shaped `[MAX_SEQUENCES, N_LAYERS, MAX_BLOCKS_PER_SEQ]`, initialized to `-1`. The index is:

```text
slot * num_layers * max_blocks_per_seq + layer * max_blocks_per_seq + logical_block
```

`logical_block = position / 16`. The value is the physical block id, or `-1` if that page has not been allocated. A host-side free stack is filled with `0 .. total_blocks-1` and popped from the back. The table lives in mapped host memory (`cudaHostAlloc` + `cudaHostAllocMapped`) so the CPU allocator and the attention kernel see the same ints without a copy. The free stack stays ordinary host memory. Only the CPU pops and pushes it.

The decode kernel indexes the table with the compile-time constants `N_LAYERS` and `MAX_BLOCKS_PER_SEQ`. Those match the host formula only while the checkpoint has 32 layers and the constants stay 32.

## Mapped memory

The engine assumes a unified GPU/CPU address space, as on Jetson. `alloc_mapped_host` allocates with `cudaHostAllocMapped` and requires `cudaHostGetDevicePointer` to return the same address. These buffers are mapped:

- prompt token ids
- `embed_proj` logits, so the CPU can sample without a device-to-host copy of the vocabulary
- the block table
- the decode token id, `gpu_seq_lens`, and `gpu_active_slots`

Weights, activations, and the KV pool are ordinary device allocations.

## What each kernel is for

A kernel is a function this repo launches on the GPU. The projections, the prefill score matrix, `scores * V`, and the LM head are the cuBLAS GEMMs in the previous section, so they are not in this list. Everything else in the forward pass is one of the kernels below. Two of them exist only so the decode benchmark can replay a CUDA graph at a new position and a new cache slot.

| Kernel | Where | What it does | Why it is there |
| --- | --- | --- | --- |
| `embeddingGatherKernel` | `prefill.cu` | Copies the embedding-table row for each token id into the hidden state. | The prompt and every new decode token arrive as integer ids. The rest of the model consumes 2048-wide vectors. A gather is an indirect lookup, so it is a kernel rather than a matrix multiply. Prefill launches one block per prompt token. Decode and the benchmark launch one block for the single new id. `embedding_gather_async` is the same kernel on a caller-chosen stream. |
| `rmsNormKernel` | `prefill.cu` and `decode.cu` | For one token, computes `x / rms(x) * weight` across all 2048 dimensions. | Llama normalizes before attention, before the MLP, and once more before the vocabulary scores. That keeps the scale of the residual stream from drifting across 32 layers. It is a reduction plus an elementwise scale, which a GEMM does not express. The two copies do the same math. Prefill synchronizes after its launch. Decode takes a stream so the benchmark can capture the step. |
| `ropeKernel` | `prefill.cu` | Rotates adjacent pairs inside Q and K by an angle that depends on the token's position in the prompt. | Q and K projections do not know which position a token occupies. RoPE puts that position into the Q·K dot product. V is left unchanged because the position signal lives in the score, not in the values being mixed. Position is the block index, so one launch covers the whole prompt. |
| `ropeKernelDecode` | `decode.cu` | The same rotation for one vector. The position is a kernel argument. | Decode rotates one token, and its position is the absolute sequence index (`current_prompt_len`), not zero. Launching the prefill kernel for that one row would treat it as position 0. `main` uses this on every decode step. |
| `ropeKernelDecodePtr` | `decode.cu` | The same rotation, but the position is loaded from a device `int` when the kernel runs. | A captured CUDA graph freezes ordinary kernel arguments. The benchmark (`src/bench.cu`) replays one graph for many positions, so the position has to be read from memory at replay time. The generation loop does not call this kernel. |
| `residualKernel` | `prefill.cu` | Adds one 2048-vector onto another, for every token in the launch. | Attention and the MLP are computed from a normalized copy of the residual stream. Their outputs are added back onto the un-normalized stream. That skip connection is how each sublayer learns an update without replacing the token's running representation. Prefill calls it after the output projection and again after the MLP down projection. |
| `residualAddKernel` | `decode.cu` | The same add, for the one decode token. | Same residual as prefill, kept in `decode.cu` so the decode step can launch it on a stream. Called after the output projection and after the MLP. |
| `siluKernel` | `prefill.cu` | Writes `silu(gate) * up` over the 8192-wide MLP intermediate, in place in the gate buffer. | Gate and up are linear projections. SiLU is the nonlinearity that makes the MLP a SwiGLU block instead of two matrix multiplies. `silu_async` is the same kernel on a stream, used by the benchmark. |
| `causalMaskKernel` | `prefill.cu` | Writes `-inf` into `scores[query, key]` wherever `key > query`. | Prefill's score GEMM is a full `[P, P]` product, so each token also receives scores against future tokens. The mask removes those scores before softmax. Decode does not launch it. The cache only contains positions up to the current token, so there is no future key to hide. |
| `softmaxKernel` | `prefill.cu` | Turns each score row into weights that sum to 1. | The following multiply with V has to be a weighted average of earlier tokens. Softmax is that conversion. Masked positions become about 0. Decode does not launch it. `pagedAttentionKernel` folds the same math into a running sum so it never stores a score per past token. |
| `pagedAttentionKernel` | `decode.cu` | For each query head of the new token, dots Q with every cached K, applies an online softmax, and accumulates the weighted V. | This is decode's replacement for the prefill score GEMM, the mask, the softmax, and `scores * V`. The keys live in pages rather than in one dense `[seq, 512]` matrix, and the sequence is one token longer every step. One block handles one query head because that head's 64 dimensions reduce to a single score. |
| `scatterKvIndirectKernel` | `decode.cu` | Copies one K or V row to `dsts[layer]`, a pointer loaded when the kernel runs. | After RoPE, K and V have to be stored or the next token cannot attend to this one. The benchmark keeps a device table of destination pointers and updates it when the token moves to a new slot in a page, so a captured graph stays valid. The generation loop in `main.cu` does the same copy with `cudaMemcpy` and does not launch this kernel. |

## Kernels

The subsections below are the launch geometry and the reductions. The table above is the reason each launch exists.

Launch bounds are hard. A block never has more than 1024 threads. Several kernels split a 2048-wide row across 1024 threads by having each thread touch `threadIdx.x` and `threadIdx.x + 1024`. Softmax and the causal mask launch `prompt_len` threads per block, so a prompt longer than 1024 tokens is skipped with a message rather than tiled. RoPE on Q launches `2048 / 2 = 1024` threads, which is exactly the limit.

### `embeddingGatherKernel`

```text
grid  = prompt_len          (decode launches 1)
block = 1024
```

Block `b` is token `b`. Thread `t` copies hidden dimensions `t` and `t + 1024` from `embed_tokens[token_id * 2048]` into the output row. The token ids are the mapped prompt buffer, or a one-element mapped buffer during decode.

### `rmsNormKernel`

```text
grid  = num_tokens
block = 1024
```

One block normalizes one token. Each thread writes the sum of squares of its two dimensions into shared memory. A tree reduction over 1024 threads sums them. Thread 0 turns that into

```text
rms = sqrt(sum_of_squares / 2048 + 1e-5)
```

Every thread then writes `x / rms * weight` for both of its dimensions. The norm weight is the layer's `input_layernorm`, `post_attention_layernorm`, or the final `model.norm`. Output is a different buffer from the input, so the residual row can stay intact.

The decode copy in `decode.cu` is the same reduction. Its bounds check is true for every thread when the grid is one block per token, which is how it is launched.

### `ropeKernel`, `ropeKernelDecode`, and `ropeKernelDecodePtr`

Llama RoPE, applied to adjacent pairs inside each head. V is not rotated. `ropeKernelDecodePtr` is the graph-replay form of `ropeKernelDecode`: the angle uses `*position` from device memory instead of a host argument baked into the launch. `src/bench.cu` calls it. `main` calls `ropeKernelDecode`.

```text
pair i in 0..31
theta = 1 / 500000 ^ (2i / 64)
angle = position * theta
(x, y) -> (x cos - y sin, x sin + y cos)
```

`pair_index = threadIdx.x % 32`, so the same 32 frequencies repeat for every head.

Prefill launches one block per token and `proj_dim / 2` threads. For Q, `proj_dim` is 2048 (32 heads). For K, it is 512 (8 heads). Position is `blockIdx.x`, which is the token's index in the prompt.

Decode launches a single block on the one vector it was given. Position is the host argument `decode_token_index`, the absolute index in the sequence, not zero. Q is rotated at `q_proj + position * 2048`. K is rotated at the start of `k_proj_batched_buffer`, because the decode K GEMM always writes row 0.

### `scatter` into pages

Prefill and the generation loop copy with `cudaMemcpy`. `scatterKvIndirectKernel` is the benchmark's version of the one-token copy: one thread per element writes `src[i]` to `dsts[layer][i]`. `dsts` is a device array of pointers, so the host can point a layer at a new page slot between graph replays.

Prefill walks the prompt in chunks of 16. For each chunk at layer `L`:

1. The block-table entry must still be `-1`. Prefill expects a fresh table.
2. Pop a physical block from the free stack and store it in the table.
3. Copy `min(16, tokens_left) * 512` BF16 values of K into the start of that block.
4. Copy the same number of V values to `block + v_offset`.

A short final page is a short copy. The rest of the page is left untouched, and attention never reads past `seq_len`.

Decode appends one token. `logical_block = position / 16` and `token_in_block = position % 16`. A new page is popped only when `token_in_block == 0` and the entry is still `-1`. Otherwise the page allocated during prefill (or by an earlier decode step) is reused. One `cudaMemcpy` of 512 BF16 values writes K, and another writes V, at `token_in_block` inside that page.

### Prefill attention scores (cuBLAS, per query head)

Q and K are row-major with heads packed in the last dimension. Head `i` of Q starts at column `i * 64` and the next token is 2048 elements later. Head `i / 4` of K has stride 512. The GEMM for head `i` is:

```text
cublasGemmEx(OP_T, OP_N,
             m = P, n = P, k = 64,
             alpha = 1/sqrt(64),
             A = K_head, lda = 512,
             B = Q_head, lda = 2048,
             C = scores_head, ldc = P)
```

The stored result is row-major `scores[query, key] = dot(Q[query], K[key]) / 8`. Grouped-query attention is the integer division `k_head = q_head / 4`: query heads 0–3 read KV head 0, heads 4–7 read KV head 1, and so on.

### `causalMaskKernel`

```text
grid  = P * 32          (one block per query row per query head)
block = P
```

`row = blockIdx.x % P` is the query position. `threadIdx.x` is the key position. If `key > query`, that score becomes `-inf` (`-HUGE_VALF` converted to BF16). Diagonal and past positions stay as the GEMM wrote them.

### `softmaxKernel`

Same launch as the mask. Each block owns one score row. In shared memory it reduces the row max, computes `exp(score - max)`, reduces the sum, and writes `exp / sum` back. Masked positions are ~0 after the exp. This is a full `[P, P]` softmax, not the online softmax used in decode.

### `scores * V` (cuBLAS, per query head)

```text
cublasGemmEx(OP_N, OP_N,
             m = 64, n = P, k = P,
             A = V_head, lda = 512,
             B = scores_head, lda = P,
             C = q_proj + head * 64, ldc = 2048)
```

The stored result is row-major attention output `[P, 2048]`, head `i` in columns `[i*64, (i+1)*64)`. Each query position is `sum_key softmax(scores)[query, key] * V[key]`, using V head `i / 4`.

### O projection and the first residual

`hidden_state = attn_out * W_o^T` through the standard GEMM, then `residualKernel`:

```text
grid = num_tokens, block = 1024
hidden[t] += residual[t]     for dims t and t+1024
```

In prefill the residual is the snapshot saved in `input_embeddings`. In decode, `decode::residualAdd` adds the O projection onto the hidden row that still holds this token's input to the layer. Same operation, two kernels.

### SwiGLU

Post-attention RMSNorm writes `rms_norms` and leaves `hidden_state` alone. Two GEMMs produce

```text
gate = rms * W_gate^T     [tokens, 8192]
up   = rms * W_up^T       [tokens, 8192]
```

`siluKernel` then overwrites gate in place:

```text
grid = num_tokens, block = 1024
for i in 0, 1024, 2048, ... 7168:
    a = gate[i]
    gate[i] = a * sigmoid(a) * up[i]
```

Each thread covers eight of the 8192 intermediate values. The down GEMM is `hidden_delta = gate * W_down^T` into `q_proj`, and a second residual add folds it into `hidden_state`.

### `pagedAttentionKernel` (decode only)

```text
grid  = dim3(num_active_slots, 32)    main launches (1, 32)
block = 64                            one thread per head dimension
```

Block `(active_slot, q_head)` produces one 64-wide output slice. `slot = gpu_active_slots[active_slot]` selects the sequence. `kv_head = q_head / 4`. The query element for this thread is `q_proj[active_slot * 2048 + q_head * 64 + thread]`, which is why decode copied Q to the front of the buffer.

The kernel walks every logical page up to `ceil(seq_len / 16)`, translates it through the block table, and walks the valid tokens in that page. For each token, the thread loads its own K dimension and V dimension from the page. The 64 threads are two warps. Each warp reduces `q[d] * k[d]` with `__shfl_down_sync` offsets 16, 8, 4, 2, 1. Thread 0 and thread 32 write the two warp sums to shared memory, thread 0 adds them and divides by `sqrt(64) == 8`, and every thread reads that scalar score.

The score updates an online softmax (the running-max form used by FlashAttention) so the full key row is never stored:

```text
m'   = max(m, score)
corr = exp(m - m')
p    = exp(score - m')
d    = d * corr + p
acc  = acc * corr + p * V[d]
m    = m'
```

`acc` and `d` start at 0 and `m` starts at `-inf`, so the first token sets `acc = p * V[d]` and `d = p`. After the last token the thread writes `acc / d` to the same Q-head slot it read. Every thread holds its own `acc` (one output dimension) and the same scalar `m` and `d`, because they all observe the same score.

This reduction is hardcoded for 64 threads in two warps. A different `HEAD_DIM` would need a different kernel.

## Prefill, layer by layer

`llama_prefill::prefill` is one function. It creates its own cuBLAS handle and synchronizes after every custom kernel.

1. Gather embeddings for all prompt tokens. Copy them into `hidden_state`. That copy is the residual stream entering layer 0.
2. For each layer:
   1. Snapshot `hidden_state` into `input_embeddings`.
   2. RMSNorm with `input_layernorm`. Copy the normalized rows back over `hidden_state` so the Q GEMM's layout comments match the buffer that is live. The GEMM itself reads `rms_norms`.
   3. Q, K, V GEMMs from `rms_norms`.
   4. RoPE on Q and on K. Positions are `0 .. P-1`.
   5. Copy K and V into new pages.
   6. Score GEMM per query head into this layer's slice of `prefill_attn_scores`.
   7. Causal mask, then softmax, over that slice.
   8. `scores * V` back into `q_proj`.
   9. O GEMM into `hidden_state`, then residual add from the snapshot.
   10. RMSNorm with `post_attention_layernorm` into `rms_norms`.
   11. Gate and up GEMMs, SiLU, down GEMM into `q_proj`, residual add into `hidden_state`.
3. Final RMSNorm with `model.norm.weight`.
4. LM-head GEMM: `logits = hidden * E^T` with `E = model.embed_tokens.weight`, written to mapped `embed_proj` shaped `[P, vocab]`.
5. Synchronize, then argmax the last row on the CPU. Print that greedy id. Record it in `generated_tokens` and `last_generated_tokens`, and set `current_prompt_len = P`.

The greedy id is a debug sample of the prefill distribution. The best-of-N loop replaces it before any decode step, and it is not written into the KV cache. Cache contents after prefill are the prompt only.

## Decode, one token

`main` keeps a single slot (`slot = 0`). After prefill it copies the last-row logits into a host vector and calls `select_diverse_first_tokens`. That builds a temperature-0.7 softmax, drops `<|end_of_text|>` (128001) and `<|eot_id|>` (128009) so a candidate cannot start by stopping, applies top-p 0.9, and takes the highest-probability remaining ids, up to `BEST_OF_N` (4). These are ranked nucleus tokens, not independent random draws.

It then snapshots the KV pool, the block table, and the free stack. Each candidate restores that snapshot, so every candidate attends to the same prompt cache and writes its own continuation on top.

For candidate `c` the RNG is `mt19937(42 + c * 10007)`. The chosen first token is pushed as already generated, with its log-prob taken from the prefill softmax. The decode loop then runs while the slot is live and fewer than `MAX_NEW_TOKENS` (64) ids have been kept.

One iteration, for the token id currently in `last_generated_tokens`, at sequence index `position = current_prompt_len`:

1. Write the id into mapped memory and gather its embedding into `hidden_state[position]`. The first time through, compare eight dimensions against the embedding table and abort on a mismatch.
2. Set `gpu_seq_lens = position + 1`. That count includes the token whose K and V this step is about to append. `gpu_active_slots[0]` stays 0.
3. For each layer:
   1. RMSNorm the one hidden row.
   2. Q GEMM into `q_proj[position]`. K and V GEMMs into row 0 of the K and V buffers.
   3. RoPE at absolute position `position`.
   4. Append that K and V into the page for `position`.
   5. Copy Q to `q_proj[0]`. Run paged attention. The output replaces Q there.
   6. O GEMM from `q_proj[0]` into `rms_norms[position]`. Residual-add onto `hidden_state[position]`.
   7. RMSNorm, gate, up, SiLU, down, residual-add. Gate and up use row `position` of the MLP buffers. Down is stored at `q_proj[position]`.
4. Final RMSNorm of this row.
5. LM-head GEMM of this one row. `n = 1`, so the logits occupy `embed_proj[0 .. vocab)`.
6. Synchronize and convert that row to FP32 on the host.
7. `sample_token`: if temperature is `<= 0`, argmax. Otherwise a FP64 softmax divided by temperature 0.7, optional top-p 0.9 with a renormalized nucleus, then one multinomial draw. The draw's log-prob under that distribution is kept for scoring.

If the draw is 128001, 128009, or `position` is already the last legal index (`511`), the slot is freed: every block-table entry for the slot goes back on the free stack and is set to `-1`. The stopping id is not appended. Otherwise the id is appended, `last_generated_tokens` is updated, and `current_prompt_len` increments by one. The next iteration embeds that new id at the next index.

Leaving the loop because `MAX_NEW_TOKENS` was hit also frees the slot. The candidate score is the mean per-token log-prob, plus `0.5` when the kept length is between 4 and 40 inclusive, and minus `2` when the length is 0, 1, or 2. The highest score wins. Its token ids are what `main` prints as `Generated tokens`.

With `BEST_OF_N == 1`, the decode loop also prints each step. With the default of 4, that per-step log is off, and the four candidates run one after another on the same slot rather than as a batched `num_active_slots`.

## Worked sequence index

Prompt length 10, one candidate, first token `T0`.

| Event | `current_prompt_len` | Cache positions | What attention sees |
| --- | --- | --- | --- |
| After prefill | 10 | `0..9` across pages | Prompt scores, masked causally. `T0` is not cached. |
| Decode step for `T0` | 10 during the step | Writes position 10, which is page 0, slot 10 | `gpu_seq_lens = 11`, keys `0..10` |
| `T0` accepted, next id is `T1` | 11 | `0..10` | |
| Decode step for `T1` | 11 during the step | Writes position 11 | 12 keys |
| Position 16 | 16 during the step | `token_in_block == 0`, pop a new page | 17 keys |

Page 0 was allocated in prefill and still has room at slots 10–15. The next page appears only when the sequence index crosses a multiple of 16.

## What is synchronized

Prefill synchronizes the device after every custom kernel launch, and again after the LM-head GEMM before the host argmax. Decode custom kernels launch on the default stream and do not synchronize themselves. Order still holds: cuBLAS and the kernels share that stream, and the host synchronizes after the LM head before it reads logits. The K and V `cudaMemcpy` calls are synchronous, so a page is fully written before `pagedAttention` is launched for that layer.

## Limits that fall out of the kernels

- One sequence. `MAX_SEQUENCES` is 1, and a full 512-token context consumes the entire block pool.
- Prompt length is at most 512, and the mask and softmax also refuse `P > 1024`.
- Decode indexing uses the sequence position as a row index into buffers sized for 512, and refuses a position at or past 512.
- Paged attention is correct for `HEAD_DIM == 64` and for the GQA ratio 4 baked into the kernel.
- The block-table index in the kernel uses compile-time `N_LAYERS == 32`.
- Mapped-pointer equality is required. A discrete GPU that returns a different device mapping fails allocation.
- Generation stops on token 128001, token 128009, 64 new tokens, or a full 512-length context.
