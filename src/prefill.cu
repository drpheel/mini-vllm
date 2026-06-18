#include "prefill.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llama_prefill {
namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
  }
}

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

void prefill(const int* gpu_input_tokens,
             size_t prompt_len,
             __nv_bfloat16* input_embeddings,
             __nv_bfloat16* hidden_state,
             __nv_bfloat16* rms_norms,
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

  embedding_gather(gpu_input_tokens, input_embeddings, weights.tok_embeddings, prompt_len);
  check_cuda(cudaMemcpy(hidden_state, input_embeddings, prompt_len * HIDDEN_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
             "cudaMemcpy(prefill hidden_state <- input_embeddings)");

  for (const __nv_bfloat16* norm_weight : weights.input_layernorm) {
    if (norm_weight == nullptr) {
      throw std::runtime_error("prefill encountered null layernorm weight");
    }
    rms_norm(hidden_state, rms_norms, norm_weight, prompt_len);
    check_cuda(cudaMemcpy(hidden_state, rms_norms, prompt_len * HIDDEN_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
               "cudaMemcpy(prefill hidden_state <- rms_norms)");
  }
}

}  // namespace llama_prefill
