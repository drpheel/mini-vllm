#pragma once

#include <cuda_bf16.h>

namespace decode {

constexpr int HIDDEN_SIZE = 2048;
constexpr int HEAD_DIM = 64;

void rmsNorm(__nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16* norm_weights, int num_tokens);
void ropeDecode(__nv_bfloat16* input, int position_in_sequence, int proj_dim, int head_dim = HEAD_DIM);

}  // namespace decode
