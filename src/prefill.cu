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

__global__ void siluKernel(__nv_bfloat16* a, const __nv_bfloat16* b) {
  const int work_index = threadIdx.x + blockIdx.x * 8192;
  for (int i = 0; i < 8192; i += 1024) {
    const float a_value = static_cast<float>(a[work_index + i]);
    const float b_value = static_cast<float>(b[work_index + i]);
    a[work_index + i] = static_cast<__nv_bfloat16>(a_value * (1.0f / (1.0f + expf(-a_value))) * b_value);
  }
}

__global__ void causalMaskKernel(__nv_bfloat16* input, int num_tokens) {
  if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS) {
    return;
  }

  const int column = threadIdx.x;
  const int row = blockIdx.x % num_tokens;
  if (column > row) {
    input[blockIdx.x * num_tokens + threadIdx.x] = __float2bfloat16(-HUGE_VALF);
  }
}

void scatter_kv_to_paged_attention_cache(const __nv_bfloat16* k_proj_batched_buffer,
                                         const __nv_bfloat16* v_proj_batched_buffer,
                                         size_t prompt_len,
                                         size_t layer,
                                         size_t num_layers,
                                         int kv_dim,
                                         PagedAttentionState* paged_attention_state) {
  if (paged_attention_state == nullptr) {
    return;
  }
  if (paged_attention_state->block_size <= 0) {
    throw std::runtime_error("paged attention requires positive block_size");
  }
  if (paged_attention_state->max_blocks_per_seq <= 0) {
    throw std::runtime_error("paged attention requires positive max_blocks_per_seq");
  }
  if (paged_attention_state->kv_cache == nullptr ||
      paged_attention_state->block_table == nullptr ||
      paged_attention_state->free_blocks == nullptr) {
    throw std::runtime_error("paged attention requires kv_cache, block_table, and free_blocks");
  }
  if (paged_attention_state->slot < 0) {
    throw std::runtime_error("paged attention requires non-negative slot");
  }
  if (num_layers == 0) {
    throw std::runtime_error("paged attention requires at least one layer");
  }
  if (paged_attention_state->block_bytes == 0) {
    throw std::runtime_error("paged attention requires positive block_bytes");
  }
  if (paged_attention_state->v_offset >= paged_attention_state->block_bytes) {
    throw std::runtime_error("paged attention requires v_offset < block_bytes");
  }

  const int prompt_len_int = static_cast<int>(prompt_len);
  for (int token_idx = 0; token_idx < prompt_len_int; token_idx += paged_attention_state->block_size) {
    int num_tokens_to_copy = prompt_len_int - token_idx;
    if (num_tokens_to_copy > paged_attention_state->block_size) {
      num_tokens_to_copy = paged_attention_state->block_size;
    }

    const int block_idx = token_idx / paged_attention_state->block_size;
    if (block_idx >= paged_attention_state->max_blocks_per_seq) {
      throw std::runtime_error("paged attention block_idx exceeds max_blocks_per_seq");
    }

    const size_t block_table_index =
        static_cast<size_t>(paged_attention_state->slot) * num_layers * paged_attention_state->max_blocks_per_seq +
        layer * paged_attention_state->max_blocks_per_seq + static_cast<size_t>(block_idx);
    int block = paged_attention_state->block_table[block_table_index];
    if (block == -1) {
      if (paged_attention_state->free_blocks_count == 0) {
        throw std::runtime_error("paged attention has no free blocks available");
      }
      const size_t free_block_idx = paged_attention_state->free_blocks_count - 1;
      block = paged_attention_state->free_blocks[free_block_idx];
      paged_attention_state->free_blocks_count = free_block_idx;
      paged_attention_state->block_table[block_table_index] = block;
    } else {
      throw std::runtime_error("paged attention expected unallocated block during prefill");
    }

    __nv_bfloat16* k_cache_ptr = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<char*>(paged_attention_state->kv_cache) + static_cast<size_t>(block) * paged_attention_state->block_bytes);
    const __nv_bfloat16* k_proj_ptr = k_proj_batched_buffer + static_cast<size_t>(token_idx) * kv_dim;
    check_cuda(cudaMemcpy(k_cache_ptr,
                          k_proj_ptr,
                          static_cast<size_t>(num_tokens_to_copy) * kv_dim * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice),
               "cudaMemcpy(prefill paged attention K)");

    __nv_bfloat16* v_cache_ptr = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<char*>(paged_attention_state->kv_cache) + static_cast<size_t>(block) * paged_attention_state->block_bytes +
        paged_attention_state->v_offset);
    const __nv_bfloat16* v_proj_ptr = v_proj_batched_buffer + static_cast<size_t>(token_idx) * kv_dim;
    check_cuda(cudaMemcpy(v_cache_ptr,
                          v_proj_ptr,
                          static_cast<size_t>(num_tokens_to_copy) * kv_dim * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice),
               "cudaMemcpy(prefill paged attention V)");
  }
}

void compute_prefill_attention_scores_gqa(cublasHandle_t cublas_handle,
                                          const __nv_bfloat16* q_proj,
                                          const __nv_bfloat16* k_proj_batched_buffer,
                                          size_t prompt_len,
                                          size_t layer,
                                          __nv_bfloat16* prefill_attn_scores) {
  if (prefill_attn_scores == nullptr) {
    return;
  }
  if (q_proj == nullptr || k_proj_batched_buffer == nullptr) {
    throw std::runtime_error("prefill attention scoring requires q_proj and k_proj buffers");
  }
  static_assert(HIDDEN_SIZE % HEAD_DIM == 0, "HIDDEN_SIZE must be divisible by HEAD_DIM");
  static_assert(KV_DIM % HEAD_DIM == 0, "KV_DIM must be divisible by HEAD_DIM");
  static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0, "NUM_Q_HEADS must be divisible by NUM_KV_HEADS");

  const int prompt_len_int = static_cast<int>(prompt_len);
  const float attn_alpha = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
  const float attn_beta = 0.0f;
  const size_t layer_offset = layer * static_cast<size_t>(NUM_Q_HEADS) * prompt_len * prompt_len;
  const size_t per_head_scores = prompt_len * prompt_len;

  for (int q_head_idx = 0; q_head_idx < NUM_Q_HEADS; ++q_head_idx) {
    const int k_head_idx = q_head_idx / GQA_Q_TO_K_RATIO;
    const __nv_bfloat16* q_head = q_proj + q_head_idx * HEAD_DIM;
    const __nv_bfloat16* k_head = k_proj_batched_buffer + k_head_idx * HEAD_DIM;
    __nv_bfloat16* attn_score_head = prefill_attn_scores + layer_offset + static_cast<size_t>(q_head_idx) * per_head_scores;

    cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    prompt_len_int,
                                                    prompt_len_int,
                                                    HEAD_DIM,
                                                    &attn_alpha,
                                                    k_head,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    q_head,
                                                    CUDA_R_16BF,
                                                    HIDDEN_SIZE,
                                                    &attn_beta,
                                                    attn_score_head,
                                                    CUDA_R_16BF,
                                                    prompt_len_int,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
    check_cublas(attn_score_status, "cublasGemmEx(prefill gqa attention scores)");
  }
}

void causal_mask(__nv_bfloat16* input, int num_tokens) {
  if (input == nullptr) {
    return;
  }
  if (num_tokens <= 0) {
    return;
  }
  if (num_tokens > 1024) {
    std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched\n";
    return;
  }

  causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
  check_cuda(cudaGetLastError(), "causalMaskKernel launch");
  check_cuda(cudaDeviceSynchronize(), "causalMaskKernel");
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

void silu(__nv_bfloat16* a, const __nv_bfloat16* b, size_t num_tokens) {
  if (num_tokens == 0) {
    return;
  }
  siluKernel<<<static_cast<unsigned int>(num_tokens), 1024>>>(a, b);
  check_cuda(cudaGetLastError(), "siluKernel launch");
  check_cuda(cudaDeviceSynchronize(), "siluKernel");
}

void prefill(const int* gpu_input_tokens,
             size_t prompt_len,
             __nv_bfloat16* input_embeddings,
             __nv_bfloat16* hidden_state,
             __nv_bfloat16* rms_norms,
             __nv_bfloat16* q_proj,
             __nv_bfloat16* k_proj_batched_buffer,
             __nv_bfloat16* v_proj_batched_buffer,
             const PrefillWeights& weights,
             PagedAttentionState* paged_attention_state,
             __nv_bfloat16* prefill_attn_scores) {
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
  if (!weights.w_k.empty() && weights.w_k.size() != weights.input_layernorm.size()) {
    throw std::runtime_error("prefill requires w_k to match input_layernorm size when k-proj is enabled");
  }
  if (!weights.w_v.empty() && weights.w_v.size() != weights.input_layernorm.size()) {
    throw std::runtime_error("prefill requires w_v to match input_layernorm size when v-proj is enabled");
  }
  if (prompt_len > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("prefill prompt_len exceeds int range for cuBLAS");
  }

  embedding_gather(gpu_input_tokens, input_embeddings, weights.tok_embeddings, prompt_len);
  check_cuda(cudaMemcpy(hidden_state, input_embeddings, prompt_len * HIDDEN_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
             "cudaMemcpy(prefill hidden_state <- input_embeddings)");

  CublasHandleGuard cublas_guard;
  if (!weights.w_q.empty() || !weights.w_k.empty() || !weights.w_v.empty()) {
    check_cublas(cublasCreate(&cublas_guard.handle), "cublasCreate(prefill)");
  }
  if (!weights.w_q.empty() && q_proj == nullptr) {
    throw std::runtime_error("prefill q-proj is enabled but q_proj buffer is null");
  }
  if (!weights.w_k.empty() && k_proj_batched_buffer == nullptr) {
    throw std::runtime_error("prefill k-proj is enabled but k_proj_batched_buffer is null");
  }
  if (!weights.w_v.empty() && v_proj_batched_buffer == nullptr) {
    throw std::runtime_error("prefill v-proj is enabled but v_proj_batched_buffer is null");
  }

  const float q_proj_alpha = 1.0f;
  const float q_proj_beta = 0.0f;
  const float k_proj_alpha = 1.0f;
  const float k_proj_beta = 0.0f;
  const float v_proj_alpha = 1.0f;
  const float v_proj_beta = 0.0f;
  const int embedding_length = HIDDEN_SIZE;
  const int kv_dim = KV_DIM;
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

      // Row-major target we want: Q = R * Wq^T, where
      //   R  = rms_norms [P, H], Wq = w_q [H, H], Q [P, H].
      // cuBLAS reads all buffers as column-major:
      //   R(row-major [P, H]) appears as R^T [H, P].
      //   Wq(row-major [H, H]) appears as Wq^T [H, H], and opA=T flips it to Wq.
      // So GEMM computes C_col = Wq * R^T = (R * Wq^T)^T.
      // Writing C in column-major gives the same bytes as row-major Q [P, H].
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

    if (!weights.w_k.empty()) {
      const __nv_bfloat16* w_k = weights.w_k[layer];
      if (w_k == nullptr) {
        throw std::runtime_error("prefill encountered null k_proj weight");
      }

      // Row-major target we want: K = R * Wk^T, where
      //   R  = rms_norms [P, H], Wk = w_k [KV, H], K [P, KV].
      // cuBLAS reads row-major buffers as column-major:
      //   R(row-major [P, H]) appears as R^T [H, P].
      //   Wk(row-major [KV, H]) appears as Wk^T [H, KV], and opA=T flips it to Wk.
      // So GEMM computes C_col = Wk * R^T = (R * Wk^T)^T.
      // Writing C in column-major gives the same bytes as row-major K [P, KV].
      cublasStatus_t k_proj_status = cublasGemmEx(cublas_guard.handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  kv_dim,
                                                  prompt_len_int,
                                                  embedding_length,
                                                  &k_proj_alpha,
                                                  w_k,
                                                  CUDA_R_16BF,
                                                  embedding_length,
                                                  rms_norms,
                                                  CUDA_R_16BF,
                                                  embedding_length,
                                                  &k_proj_beta,
                                                  k_proj_batched_buffer,
                                                  CUDA_R_16BF,
                                                  kv_dim,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);
      check_cublas(k_proj_status, "cublasGemmEx(prefill k_proj)");
    }

    if (!weights.w_v.empty()) {
      const __nv_bfloat16* w_v = weights.w_v[layer];
      if (w_v == nullptr) {
        throw std::runtime_error("prefill encountered null v_proj weight");
      }

      // Row-major target we want: V = R * Wv^T, where
      //   R  = rms_norms [P, H], Wv = w_v [KV, H], V [P, KV].
      // cuBLAS sees R as R^T and sees Wv as Wv^T, then opA=T flips Wv back.
      // So GEMM computes C_col = Wv * R^T = (R * Wv^T)^T.
      // Writing C in column-major gives row-major V [P, KV] in this buffer.
      cublasStatus_t v_proj_status = cublasGemmEx(cublas_guard.handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  kv_dim,
                                                  prompt_len_int,
                                                  embedding_length,
                                                  &v_proj_alpha,
                                                  w_v,
                                                  CUDA_R_16BF,
                                                  embedding_length,
                                                  rms_norms,
                                                  CUDA_R_16BF,
                                                  embedding_length,
                                                  &v_proj_beta,
                                                  v_proj_batched_buffer,
                                                  CUDA_R_16BF,
                                                  kv_dim,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);
      check_cublas(v_proj_status, "cublasGemmEx(prefill v_proj)");
    }
    
    if (!weights.w_q.empty()) {
      rope(q_proj, prompt_len, embedding_length);
    }
    if (!weights.w_k.empty()) {
      rope(k_proj_batched_buffer, prompt_len, kv_dim);
    }

    if (!weights.w_k.empty() && !weights.w_v.empty()) {
      scatter_kv_to_paged_attention_cache(
          k_proj_batched_buffer, v_proj_batched_buffer, prompt_len, layer, weights.input_layernorm.size(), kv_dim,
          paged_attention_state);
    }

    if (!weights.w_q.empty() && !weights.w_k.empty()) {
      compute_prefill_attention_scores_gqa(
          cublas_guard.handle, q_proj, k_proj_batched_buffer, prompt_len, layer, prefill_attn_scores);
    }
    
  }

  if (prefill_attn_scores != nullptr) {
    const size_t per_layer_scores = static_cast<size_t>(NUM_Q_HEADS) * prompt_len * prompt_len;
    for (size_t layer = 0; layer < weights.input_layernorm.size(); ++layer) {
      causal_mask(prefill_attn_scores + layer * per_layer_scores, prompt_len_int);
    }
  }
}

}  // namespace llama_prefill
