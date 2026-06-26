#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <vector>

namespace llama_prefill {

constexpr int HIDDEN_SIZE = 2048;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = HIDDEN_SIZE / HEAD_DIM;
constexpr int NUM_KV_HEADS = KV_DIM / HEAD_DIM;
constexpr int GQA_Q_TO_K_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;

struct PrefillWeights {
  __nv_bfloat16* tok_embeddings = nullptr;
  std::vector<__nv_bfloat16*> input_layernorm;
  std::vector<__nv_bfloat16*> w_q;
  std::vector<__nv_bfloat16*> w_k;
  std::vector<__nv_bfloat16*> w_v;
  std::vector<__nv_bfloat16*> w_o;
};

struct PagedAttentionState {
  void* kv_cache = nullptr;
  int* block_table = nullptr;
  int* free_blocks = nullptr;
  size_t free_blocks_count = 0;
  int slot = 0;
  int max_blocks_per_seq = 0;
  int block_size = 0;
  size_t block_bytes = 0;
  size_t v_offset = 0;
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

void silu(__nv_bfloat16* a, const __nv_bfloat16* b, size_t num_tokens);

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
             __nv_bfloat16* prefill_attn_scores);

}  // namespace llama_prefill
