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

__global__ void ropeKernelDecode(__nv_bfloat16* input, int position_in_sequence, int proj_dim, int head_dim) {
  if (2 * threadIdx.x + 1 >= proj_dim) {
    return;
  }

  const int pair_index = threadIdx.x % (head_dim / 2);
  const int double_i = 2 * pair_index;
  const float theta = 1.0f / powf(500000.0f, static_cast<float>(double_i) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(position_in_sequence) * theta;
  const __nv_bfloat16 prev_2i = input[2 * threadIdx.x];
  const __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1];
  input[2 * threadIdx.x] =
      static_cast<__nv_bfloat16>(static_cast<float>(prev_2i) * cosf(angle) - static_cast<float>(prev_2i_1) * sinf(angle));
  input[2 * threadIdx.x + 1] =
      static_cast<__nv_bfloat16>(static_cast<float>(prev_2i) * sinf(angle) + static_cast<float>(prev_2i_1) * cosf(angle));
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

void ropeDecode(__nv_bfloat16* input, int position_in_sequence, int proj_dim, int head_dim) {
  if (proj_dim <= 0 || proj_dim % 2 != 0) {
    return;
  }
  if (head_dim <= 0 || head_dim % 2 != 0) {
    return;
  }

  const int num_threads = proj_dim / 2;
  if (num_threads > 1024) {
    std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched\n";
    return;
  }

  ropeKernelDecode<<<1, num_threads>>>(input, position_in_sequence, proj_dim, head_dim);
#ifdef DEBUG
  cudaError error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
  }
#endif
}

}  // namespace decode
