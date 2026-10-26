# CUDA Llama 3.2 1B engine

C++/CUDA inference for Llama 3.2 1B Instruct. Weights load from a BF16 safetensors file. cuBLAS runs the projections. Custom kernels run embeddings, RMSNorm, RoPE, causal masking, softmax, SwiGLU, residuals, and paged attention during decode.

How the passes fit together, including what each kernel computes and how prefill fills the KV cache that decode reads, is written up in [docs/engine.md](docs/engine.md).

## Build

```bash
cmake --preset default
cmake --build --preset default
./build/cuda_cublas_json /path/to/model.safetensors /path/to/prompt_tokens.txt
```

The prompt file is space-separated or comma-separated token ids. `tools/tokenizer.py` can produce that file from text.

If you prefer not to use presets:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build
./build/cuda_cublas_json
```

On Jetson boards, you can optionally pass a specific architecture:

```bash
cmake --preset default -DCMAKE_CUDA_ARCHITECTURES=87
```

Use `87` for many Jetson Orin devices and `72` for many Jetson Xavier devices.

The default model path when you omit arguments is `/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors`. Mapped host memory is required for logits, the block table, and token ids, which matches Jetson unified memory.
