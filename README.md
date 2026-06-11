# CUDA cuBLAS JSON Starter

Minimal C++/CUDA project wired for:

- CUDA runtime
- cuBLAS
- `nlohmann::json`

The sample program reads a small JSON config, fills two vectors on the GPU with a CUDA kernel, and calls `cublasSaxpy` to compute:

```text
y = alpha * x + y
```

## Build

```bash
cmake --preset default
cmake --build --preset default
./build/cuda_cublas_json
```

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
