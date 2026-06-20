#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <vector>

namespace llama_prefill {

constexpr int HIDDEN_SIZE = 2048;
constexpr int KV_DIM = 512;

struct PrefillWeights {
  __nv_bfloat16* tok_embeddings = nullptr;
  std::vector<__nv_bfloat16*> input_layernorm;
  std::vector<__nv_bfloat16*> w_q;
  std::vector<__nv_bfloat16*> w_k;
  std::vector<__nv_bfloat16*> w_v;
};

void embedding_gather(const int* gpu_input_tokens,
                      __nv_bfloat16* input_embeddings,
                      const __nv_bfloat16* embed_tokens,
                      size_t prompt_len);

void rms_norm(const __nv_bfloat16* input,
              __nv_bfloat16* output,
              const __nv_bfloat16* norm_weights,
              size_t prompt_len);

void rope(__nv_bfloat16* input, size_t num_tokens, int proj_dim, int head_dim = 64);

void residual_add(__nv_bfloat16* input, const __nv_bfloat16* input_embeds, size_t num_tokens);

void prefill(const int* gpu_input_tokens,
             size_t prompt_len,
             __nv_bfloat16* input_embeddings,
             __nv_bfloat16* hidden_state,
             __nv_bfloat16* rms_norms,
             __nv_bfloat16* q_proj,
             __nv_bfloat16* k_proj_batched_buffer,
             __nv_bfloat16* v_proj_batched_buffer,
             const PrefillWeights& weights);

}  // namespace llama_prefill
