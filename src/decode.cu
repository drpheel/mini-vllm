#include "decode.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

namespace decode {
namespace {

__global__ void rmsNormKernel(__nv_bfloat16* input,
                              __nv_bfloat16* output,
                              __nv_bfloat16* norm_weights,
                              int num_tokens) {
  __shared__ float rms_vector[1024];
  const int work_index = threadIdx.x + blockIdx.x * HIDDEN_SIZE;
  if (work_index < num_tokens * HIDDEN_SIZE) {
    rms_vector[threadIdx.x] =
        static_cast<float>(input[work_index]) * static_cast<float>(input[work_index]) +
        static_cast<float>(input[work_index + 1024]) * static_cast<float>(input[work_index + 1024]);
    __syncthreads();

    for (int i = 1; i < 1024; i *= 2) {
      if (threadIdx.x % (i * 2) == 0) {
        rms_vector[threadIdx.x] += rms_vector[threadIdx.x + i];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      rms_vector[0] = sqrtf(rms_vector[0] / static_cast<float>(HIDDEN_SIZE) + 1.0e-5f);
    }
    __syncthreads();

    output[work_index] = static_cast<__nv_bfloat16>(
        (static_cast<float>(input[work_index]) / rms_vector[0]) * static_cast<float>(norm_weights[threadIdx.x]));
    output[work_index + 1024] = static_cast<__nv_bfloat16>(
        (static_cast<float>(input[work_index + 1024]) / rms_vector[0]) * static_cast<float>(norm_weights[threadIdx.x + 1024]));
  }
}

}  // namespace

void rmsNorm(__nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16* norm_weights, int num_tokens) {
  if (num_tokens <= 0) {
    return;
  }
  rmsNormKernel<<<num_tokens, 1024>>>(input, output, norm_weights, num_tokens);
#ifdef DEBUG
  cudaError error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
  }
#endif
}

}  // namespace decode
