#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <vector>

namespace llama_prefill {

constexpr int HIDDEN_SIZE = 2048;

struct PrefillWeights {
  __nv_bfloat16* tok_embeddings = nullptr;
  std::vector<__nv_bfloat16*> input_layernorm;
};

void embedding_gather(const int* gpu_input_tokens,
                      __nv_bfloat16* input_embeddings,
                      const __nv_bfloat16* embed_tokens,
                      size_t prompt_len);

void rms_norm(const __nv_bfloat16* input,
              __nv_bfloat16* output,
              const __nv_bfloat16* norm_weights,
              size_t prompt_len);

void prefill(const int* gpu_input_tokens,
             size_t prompt_len,
             __nv_bfloat16* input_embeddings,
             __nv_bfloat16* hidden_state,
             __nv_bfloat16* rms_norms,
             const PrefillWeights& weights);

}  // namespace llama_prefill
