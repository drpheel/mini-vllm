#include "prefill.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace llama_prefill {
namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* call) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(call) + " failed with cublas status " + std::to_string(static_cast<int>(status)));
  }
}

struct CublasHandleGuard {
  cublasHandle_t handle = nullptr;

  ~CublasHandleGuard() {
    if (handle != nullptr) {
      (void)cublasDestroy(handle);
    }
  }
};

__global__ void embeddingGatherKernel(const int* gpu_input_tokens,
                                      __nv_bfloat16* input_embeddings,
                                      const __nv_bfloat16* embed_tokens) {
  const int work_index = threadIdx.x + blockIdx.x * HIDDEN_SIZE;
  input_embeddings[work_index] = embed_tokens[gpu_input_tokens[blockIdx.x] * HIDDEN_SIZE + threadIdx.x];
  input_embeddings[work_index + 1024] =
      embed_tokens[gpu_input_tokens[blockIdx.x] * HIDDEN_SIZE + threadIdx.x + 1024];
}

__global__ void rmsNormKernel(const __nv_bfloat16* input, __nv_bfloat16* output, const __nv_bfloat16* norm_weights) {
  __shared__ float rms_vector[1024];
  const int work_index = threadIdx.x + blockIdx.x * HIDDEN_SIZE;
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

  output[work_index] =
      static_cast<__nv_bfloat16>((static_cast<float>(input[work_index]) / rms_vector[0]) * static_cast<float>(norm_weights[threadIdx.x]));
  output[work_index + 1024] = static_cast<__nv_bfloat16>(
      (static_cast<float>(input[work_index + 1024]) / rms_vector[0]) * static_cast<float>(norm_weights[threadIdx.x + 1024]));
}

__global__ void ropeKernel(__nv_bfloat16* input, int num_tokens, int proj_dim, int head_dim) {
  const int idx = 2 * threadIdx.x + blockIdx.x * proj_dim;
  if (idx + 1 >= num_tokens * proj_dim) {
    return;
  }

  const int pair_index = threadIdx.x % (head_dim / 2);
  const int double_i = 2 * pair_index;
  const float theta = 1.0f / powf(500000.0f, static_cast<float>(double_i) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(blockIdx.x) * theta;
  const __nv_bfloat16 prev_2i = input[idx];
  const __nv_bfloat16 prev_2i_1 = input[idx + 1];
  input[idx] = static_cast<__nv_bfloat16>(static_cast<float>(prev_2i) * cosf(angle) - static_cast<float>(prev_2i_1) * sinf(angle));
  input[idx + 1] = static_cast<__nv_bfloat16>(static_cast<float>(prev_2i) * sinf(angle) + static_cast<float>(prev_2i_1) * cosf(angle));
}

__global__ void residualKernel(__nv_bfloat16* input, const __nv_bfloat16* input_embeds) {
  const int work_index = threadIdx.x + blockIdx.x * HIDDEN_SIZE;
  input[work_index] = static_cast<__nv_bfloat16>(static_cast<float>(input[work_index]) + static_cast<float>(input_embeds[work_index]));
  input[work_index + 1024] =
      static_cast<__nv_bfloat16>(static_cast<float>(input[work_index + 1024]) + static_cast<float>(input_embeds[work_index + 1024]));
}

}  // namespace

void embedding_gather(const int* gpu_input_tokens,
                      __nv_bfloat16* input_embeddings,
                      const __nv_bfloat16* embed_tokens,
                      size_t prompt_len) {
  if (prompt_len == 0) {
    return;
  }

  embeddingGatherKernel<<<static_cast<unsigned int>(prompt_len), 1024>>>(gpu_input_tokens, input_embeddings, embed_tokens);
  check_cuda(cudaGetLastError(), "embeddingGatherKernel launch");
  check_cuda(cudaDeviceSynchronize(), "embeddingGatherKernel");
}

void rms_norm(const __nv_bfloat16* input,
              __nv_bfloat16* output,
              const __nv_bfloat16* norm_weights,
              size_t prompt_len) {
  if (prompt_len == 0) {
    return;
  }

  rmsNormKernel<<<static_cast<unsigned int>(prompt_len), 1024>>>(input, output, norm_weights);
  check_cuda(cudaGetLastError(), "rmsNormKernel launch");
  check_cuda(cudaDeviceSynchronize(), "rmsNormKernel");
}

void rope(__nv_bfloat16* input, size_t num_tokens, int proj_dim, int head_dim) {
  if (num_tokens == 0) {
    return;
  }
  if (proj_dim <= 0 || proj_dim % 2 != 0) {
    throw std::runtime_error("rope requires an even, positive proj_dim");
  }
  if (head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("rope requires an even, positive head_dim");
  }

  const int num_threads = proj_dim / 2;
  if (num_threads > 1024) {
    std::cout << "Can't launch more than 1024 threads, RoPE kernel not launched\n";
    return;
  }

  ropeKernel<<<static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(num_threads)>>>(
      input, static_cast<int>(num_tokens), proj_dim, head_dim);
  check_cuda(cudaGetLastError(), "ropeKernel launch");
  check_cuda(cudaDeviceSynchronize(), "ropeKernel");
}

void residual_add(__nv_bfloat16* input, const __nv_bfloat16* input_embeds, size_t num_tokens) {
  if (num_tokens == 0) {
    return;
  }
  residualKernel<<<static_cast<unsigned int>(num_tokens), 1024>>>(input, input_embeds);
  check_cuda(cudaGetLastError(), "residualKernel launch");
  check_cuda(cudaDeviceSynchronize(), "residualKernel");
}

void prefill(const int* gpu_input_tokens,
             size_t prompt_len,
             __nv_bfloat16* input_embeddings,
             __nv_bfloat16* hidden_state,
             __nv_bfloat16* rms_norms,
             __nv_bfloat16* q_proj,
             const PrefillWeights& weights) {
  if (prompt_len == 0) {
    return;
  }
  if (weights.tok_embeddings == nullptr) {
    throw std::runtime_error("prefill requires tok_embeddings");
  }
  if (weights.input_layernorm.empty()) {
    throw std::runtime_error("prefill requires at least one input layernorm weight");
  }
  if (!weights.w_q.empty() && weights.w_q.size() != weights.input_layernorm.size()) {
    throw std::runtime_error("prefill requires w_q to match input_layernorm size when q-proj is enabled");
  }
  if (prompt_len > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("prefill prompt_len exceeds int range for cuBLAS");
  }

  embedding_gather(gpu_input_tokens, input_embeddings, weights.tok_embeddings, prompt_len);
  check_cuda(cudaMemcpy(hidden_state, input_embeddings, prompt_len * HIDDEN_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
             "cudaMemcpy(prefill hidden_state <- input_embeddings)");

  CublasHandleGuard cublas_guard;
  if (!weights.w_q.empty()) {
    if (q_proj == nullptr) {
      throw std::runtime_error("prefill q-proj is enabled but q_proj buffer is null");
    }
    check_cublas(cublasCreate(&cublas_guard.handle), "cublasCreate(prefill)");
  }

  const float q_proj_alpha = 1.0f;
  const float q_proj_beta = 0.0f;
  const int embedding_length = HIDDEN_SIZE;
  const int prompt_len_int = static_cast<int>(prompt_len);

  for (size_t layer = 0; layer < weights.input_layernorm.size(); ++layer) {
    const __nv_bfloat16* norm_weight = weights.input_layernorm[layer];
    if (norm_weight == nullptr) {
      throw std::runtime_error("prefill encountered null layernorm weight");
    }
    rms_norm(hidden_state, rms_norms, norm_weight, prompt_len);
    check_cuda(cudaMemcpy(hidden_state, rms_norms, prompt_len * HIDDEN_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
               "cudaMemcpy(prefill hidden_state <- rms_norms)");

    if (!weights.w_q.empty()) {
      const __nv_bfloat16* w_q = weights.w_q[layer];
      if (w_q == nullptr) {
        throw std::runtime_error("prefill encountered null q_proj weight");
      }

      // We conceptually want Q = inputs * wq^T in row-major form.
      // cublasGemmEx interprets buffers as column-major, so feeding row-major A/B
      // effectively means GEMM computes and stores C^T for our row-major C.
      // With this operand order/transposition setup, the output buffer already
      // corresponds to the transposed result we need, so no extra transpose pass.
      cublasStatus_t q_proj_status = cublasGemmEx(cublas_guard.handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  embedding_length,
                                                  prompt_len_int,
                                                  embedding_length,
                                                  &q_proj_alpha,
                                                  w_q,
                                                  CUDA_R_16BF,
                                                  embedding_length,
                                                  rms_norms,
                                                  CUDA_R_16BF,
                                                  embedding_length,
                                                  &q_proj_beta,
                                                  q_proj,
                                                  CUDA_R_16BF,
                                                  embedding_length,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);
      check_cublas(q_proj_status, "cublasGemmEx(prefill q_proj)");
    }
  }
}

}  // namespace llama_prefill
