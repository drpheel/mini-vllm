#pragma once

// Llama 3.2 RoPE. theta is 500000. Frequencies whose wavelength is longer than the
// original 8192-token context are divided by 32, with a smooth band down to wavelength
// 2048. High frequencies are unchanged, including at short positions.
// https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py

__host__ __device__ inline float llama3_inv_freq(int pair_index, int head_dim) {
  const float expo = static_cast<float>(2 * pair_index) / static_cast<float>(head_dim);
  const float inv_freq = 1.0f / powf(500000.0f, expo);
  constexpr float kFactor = 32.0f;
  constexpr float kLowFreqFactor = 1.0f;
  constexpr float kHighFreqFactor = 4.0f;
  constexpr float kOriginalContext = 8192.0f;
  constexpr float kTwoPi = 6.283185307179586f;
  const float wavelen = kTwoPi / inv_freq;
  const float low_freq_wavelen = kOriginalContext / kLowFreqFactor;
  const float high_freq_wavelen = kOriginalContext / kHighFreqFactor;
  if (wavelen > low_freq_wavelen) {
    return inv_freq / kFactor;
  }
  if (wavelen < high_freq_wavelen) {
    return inv_freq;
  }
  const float smooth = (kOriginalContext / wavelen - kLowFreqFactor) / (kHighFreqFactor - kLowFreqFactor);
  return (1.0f - smooth) * (inv_freq / kFactor) + smooth * inv_freq;
}

// Llama rotate-half: dimension i is paired with i + head_dim/2, not with i+1.
__host__ __device__ inline void llama_rope_offsets(int pair_in_row, int head_dim, int* pair_index, int* low, int* high) {
  const int half = head_dim / 2;
  const int head = pair_in_row / half;
  *pair_index = pair_in_row - head * half;
  *low = head * head_dim + *pair_index;
  *high = *low + half;
}
