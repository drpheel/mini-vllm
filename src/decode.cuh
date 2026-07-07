#pragma once

#include <cuda_bf16.h>

namespace decode {

constexpr int HIDDEN_SIZE = 2048;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = HIDDEN_SIZE / HEAD_DIM;
constexpr int NUM_KV_HEADS = KV_DIM / HEAD_DIM;
constexpr int GQA_Q_TO_K_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;
constexpr int BLOCK_SIZE = 16;
constexpr int MAX_BLOCKS_PER_SEQ = 32;
constexpr int N_LAYERS = 32;
constexpr int BLOCK_BYTES = BLOCK_SIZE * KV_DIM * static_cast<int>(sizeof(__nv_bfloat16)) * 2;
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * static_cast<int>(sizeof(__nv_bfloat16));
constexpr float SQRT_HEAD_DIM = 8.0f;

void rmsNorm(__nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16* norm_weights, int num_tokens);
void ropeDecode(__nv_bfloat16* input, int position_in_sequence, int proj_dim, int head_dim = HEAD_DIM);
void pagedAttention(int layer,
                    int num_active_slots,
                    __nv_bfloat16* q_proj,
                    __nv_bfloat16* kv_cache,
                    int* block_table_gpu,
                    int* gpu_seq_lens,
                    int* gpu_active_slots,
                    __nv_bfloat16* output);
void residualAdd(__nv_bfloat16* hidden_state, const __nv_bfloat16* o_proj, int num_active_slots);

}  // namespace decode
