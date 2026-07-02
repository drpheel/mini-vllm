#pragma once

#include <cuda_bf16.h>

namespace decode {

constexpr int HIDDEN_SIZE = 2048;

void rmsNorm(__nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16* norm_weights, int num_tokens);

}  // namespace decode
