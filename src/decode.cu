#include "decode.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <iostream>

namespace decode {
namespace {

constexpr int EMBEDDING_LENGTH = HIDDEN_SIZE;

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

// Inside a single particular thread that processes a single position of particular Q head for a
// particular sequence, for particular layer.
__global__ void pagedAttentionKernel(int layer,
                                     int num_active_slots,
                                     __nv_bfloat16* q_proj,
                                     __nv_bfloat16* kv_cache,
                                     int* block_table_gpu,
                                     int* gpu_seq_lens,
                                     int* gpu_active_slots,
                                     __nv_bfloat16* output) {
  __shared__ float dot_products[2];
  const int active_slot = blockIdx.x;
  const int slot = gpu_active_slots[active_slot];
  const int q_head_id = blockIdx.y;
  const int thread_id = threadIdx.x;
  const int kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO;
  const __nv_bfloat16 q = q_proj[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id];
  const int seq_len = gpu_seq_lens[active_slot];
  const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Online softmax: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
  float current_max = -INFINITY;
  float acc = 0.0f;
  float d = 0.0f;

  for (int logical_block_idx = 0; logical_block_idx < num_blocks; ++logical_block_idx) {
    const int physical_block =
        block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
    const int tokens_in_block = min(seq_len - logical_block_idx * BLOCK_SIZE, BLOCK_SIZE);
    for (int token = 0; token < tokens_in_block; ++token) {
      __nv_bfloat16* k = (__nv_bfloat16*)((char*)kv_cache + physical_block * BLOCK_BYTES +
                                          token * KV_DIM * sizeof(__nv_bfloat16) +
                                          kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
      __nv_bfloat16* v = (__nv_bfloat16*)((char*)kv_cache + physical_block * BLOCK_BYTES + V_OFFSET +
                                          token * KV_DIM * sizeof(__nv_bfloat16) +
                                          kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
      float qk = static_cast<float>(q) * static_cast<float>(*k);
      qk += __shfl_down_sync(0xffffffff, qk, 16);
      qk += __shfl_down_sync(0xffffffff, qk, 8);
      qk += __shfl_down_sync(0xffffffff, qk, 4);
      qk += __shfl_down_sync(0xffffffff, qk, 2);
      qk += __shfl_down_sync(0xffffffff, qk, 1);
      if (thread_id == 0) {
        dot_products[0] = qk;
      }
      if (thread_id == 32) {
        dot_products[1] = qk;
      }
      __syncthreads();
      if (thread_id == 0) {
        dot_products[0] = (dot_products[0] + dot_products[1]) / SQRT_HEAD_DIM;
      }
      __syncthreads();
      const float dot_product = dot_products[0];

      float new_max = current_max;
      if (dot_product > current_max) {
        new_max = dot_product;
      }
      const float correction_factor = expf(current_max - new_max);
      current_max = new_max;
      const float exp_score = expf(dot_product - current_max);
      d = d * correction_factor + exp_score;
      acc = acc * correction_factor + exp_score * static_cast<float>(*v);
    }
  }
  output[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id] = acc / d;
}

__global__ void residualAddKernel(__nv_bfloat16* hidden_state, const __nv_bfloat16* o_proj) {
  const int work_index = threadIdx.x + blockIdx.x * HIDDEN_SIZE;
  hidden_state[work_index] =
      static_cast<__nv_bfloat16>(static_cast<float>(hidden_state[work_index]) + static_cast<float>(o_proj[work_index]));
  hidden_state[work_index + 1024] = static_cast<__nv_bfloat16>(static_cast<float>(hidden_state[work_index + 1024]) +
                                                               static_cast<float>(o_proj[work_index + 1024]));
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

void pagedAttention(int layer,
                    int num_active_slots,
                    __nv_bfloat16* q_proj,
                    __nv_bfloat16* kv_cache,
                    int* block_table_gpu,
                    int* gpu_seq_lens,
                    int* gpu_active_slots,
                    __nv_bfloat16* output) {
  if (num_active_slots <= 0) {
    return;
  }
  pagedAttentionKernel<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>(
      layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, output);
#ifdef DEBUG
  cudaError error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
  }
#endif
}

void residualAdd(__nv_bfloat16* hidden_state, const __nv_bfloat16* o_proj, int num_active_slots) {
  if (num_active_slots <= 0) {
    return;
  }
  residualAddKernel<<<num_active_slots, 1024>>>(hidden_state, o_proj);
#ifdef DEBUG
  cudaError error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
  }
#endif
}

}  // namespace decode
