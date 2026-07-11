#include "../src/prefill.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
  }
}

float rope_theta_for_pair(int pair_index, int head_dim) {
  const int double_i = 2 * pair_index;
  return 1.0f / std::pow(500000.0f, static_cast<float>(double_i) / static_cast<float>(head_dim));
}

int run_prefill_test() {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "Skipping prefill test: no CUDA device available\n";
    return 77;
  }

  constexpr int vocab_size = 4;
  constexpr size_t prompt_len = 2;
  const std::vector<int> prompt_tokens{1, 3};
  const size_t hidden_bytes = prompt_len * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);
  const size_t kv_proj_bytes = prompt_len * llama_prefill::KV_DIM * sizeof(__nv_bfloat16);
  const size_t embed_bytes = static_cast<size_t>(vocab_size) * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);
  const size_t w_q_bytes = static_cast<size_t>(llama_prefill::HIDDEN_SIZE) * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);
  const size_t w_k_bytes = static_cast<size_t>(llama_prefill::KV_DIM) * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);
  const size_t w_v_bytes = static_cast<size_t>(llama_prefill::KV_DIM) * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);
  const size_t w_o_bytes = static_cast<size_t>(llama_prefill::HIDDEN_SIZE) * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);
  const size_t w_gate_bytes =
      static_cast<size_t>(llama_prefill::MLP_INTERMEDIATE_SIZE) * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);
  const size_t w_up_bytes = w_gate_bytes;
  const size_t w_down_bytes = w_gate_bytes;
  const size_t mlp_intermediate_bytes = prompt_len * llama_prefill::MLP_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16);
  const size_t prefill_scores_elements =
      static_cast<size_t>(llama_prefill::NUM_Q_HEADS) * prompt_len * prompt_len;
  const size_t prefill_scores_bytes = std::max(prefill_scores_elements * sizeof(__nv_bfloat16), sizeof(__nv_bfloat16));
  const size_t embed_proj_bytes = prompt_len * static_cast<size_t>(vocab_size) * sizeof(__nv_bfloat16);

  std::vector<__nv_bfloat16> embed_cpu(static_cast<size_t>(vocab_size) * llama_prefill::HIDDEN_SIZE, __float2bfloat16(1.0f));
  std::vector<__nv_bfloat16> layernorm_cpu(llama_prefill::HIDDEN_SIZE, __float2bfloat16(1.0f));
  std::vector<__nv_bfloat16> w_q_cpu(static_cast<size_t>(llama_prefill::HIDDEN_SIZE) * llama_prefill::HIDDEN_SIZE,
                                     __float2bfloat16(0.0f));
  std::vector<__nv_bfloat16> w_k_cpu(static_cast<size_t>(llama_prefill::KV_DIM) * llama_prefill::HIDDEN_SIZE,
                                     __float2bfloat16(0.0f));
  std::vector<__nv_bfloat16> w_v_cpu(static_cast<size_t>(llama_prefill::KV_DIM) * llama_prefill::HIDDEN_SIZE,
                                     __float2bfloat16(0.0f));
  std::vector<__nv_bfloat16> w_o_cpu(static_cast<size_t>(llama_prefill::HIDDEN_SIZE) * llama_prefill::HIDDEN_SIZE,
                                     __float2bfloat16(0.0f));
  std::vector<__nv_bfloat16> w_gate_cpu(
      static_cast<size_t>(llama_prefill::MLP_INTERMEDIATE_SIZE) * llama_prefill::HIDDEN_SIZE, __float2bfloat16(0.0f));
  std::vector<__nv_bfloat16> w_up_cpu(
      static_cast<size_t>(llama_prefill::MLP_INTERMEDIATE_SIZE) * llama_prefill::HIDDEN_SIZE, __float2bfloat16(0.0f));
  std::vector<__nv_bfloat16> w_down_cpu(
      static_cast<size_t>(llama_prefill::HIDDEN_SIZE) * llama_prefill::MLP_INTERMEDIATE_SIZE, __float2bfloat16(0.0f));

  int* gpu_prompt_tokens = nullptr;
  __nv_bfloat16* gpu_embed_tokens = nullptr;
  __nv_bfloat16* gpu_input_embeddings = nullptr;
  __nv_bfloat16* gpu_hidden_state = nullptr;
  __nv_bfloat16* gpu_rms_norms = nullptr;
  __nv_bfloat16* gpu_q_proj = nullptr;
  __nv_bfloat16* gpu_k_proj_batched_buffer = nullptr;
  __nv_bfloat16* gpu_v_proj_batched_buffer = nullptr;
  __nv_bfloat16* gpu_w_q = nullptr;
  __nv_bfloat16* gpu_w_k = nullptr;
  __nv_bfloat16* gpu_w_v = nullptr;
  __nv_bfloat16* gpu_w_o = nullptr;
  __nv_bfloat16* gpu_w_gate = nullptr;
  __nv_bfloat16* gpu_w_up = nullptr;
  __nv_bfloat16* gpu_w_down = nullptr;
  __nv_bfloat16* gpu_mlp_gate = nullptr;
  __nv_bfloat16* gpu_mlp_up = nullptr;
  __nv_bfloat16* gpu_prefill_attn_scores = nullptr;
  __nv_bfloat16* gpu_embed_proj = nullptr;
  __nv_bfloat16* gpu_layernorm_weights = nullptr;
  __nv_bfloat16* gpu_ffn_layernorm_weights = nullptr;
  __nv_bfloat16* gpu_residual_input = nullptr;
  __nv_bfloat16* gpu_residual_embeds = nullptr;

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_prompt_tokens), prompt_tokens.size() * sizeof(int)),
             "cudaMalloc(gpu_prompt_tokens)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_embed_tokens), embed_bytes), "cudaMalloc(gpu_embed_tokens)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_input_embeddings), hidden_bytes), "cudaMalloc(gpu_input_embeddings)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_hidden_state), hidden_bytes), "cudaMalloc(gpu_hidden_state)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_rms_norms), hidden_bytes), "cudaMalloc(gpu_rms_norms)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_q_proj), hidden_bytes), "cudaMalloc(gpu_q_proj)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_k_proj_batched_buffer), kv_proj_bytes),
             "cudaMalloc(gpu_k_proj_batched_buffer)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_v_proj_batched_buffer), kv_proj_bytes),
             "cudaMalloc(gpu_v_proj_batched_buffer)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_w_q), w_q_bytes), "cudaMalloc(gpu_w_q)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_w_k), w_k_bytes), "cudaMalloc(gpu_w_k)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_w_v), w_v_bytes), "cudaMalloc(gpu_w_v)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_w_o), w_o_bytes), "cudaMalloc(gpu_w_o)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_w_gate), w_gate_bytes), "cudaMalloc(gpu_w_gate)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_w_up), w_up_bytes), "cudaMalloc(gpu_w_up)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_w_down), w_down_bytes), "cudaMalloc(gpu_w_down)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_mlp_gate), mlp_intermediate_bytes), "cudaMalloc(gpu_mlp_gate)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_mlp_up), mlp_intermediate_bytes), "cudaMalloc(gpu_mlp_up)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_prefill_attn_scores), prefill_scores_bytes),
             "cudaMalloc(gpu_prefill_attn_scores)");
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&gpu_embed_proj), embed_proj_bytes, cudaHostAllocMapped),
             "cudaHostAlloc(gpu_embed_proj)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_layernorm_weights), layernorm_cpu.size() * sizeof(__nv_bfloat16)),
             "cudaMalloc(gpu_layernorm_weights)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_ffn_layernorm_weights), layernorm_cpu.size() * sizeof(__nv_bfloat16)),
             "cudaMalloc(gpu_ffn_layernorm_weights)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_residual_input), hidden_bytes), "cudaMalloc(gpu_residual_input)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_residual_embeds), hidden_bytes), "cudaMalloc(gpu_residual_embeds)");

  check_cuda(cudaMemcpy(gpu_prompt_tokens, prompt_tokens.data(), prompt_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
             "cudaMemcpy(prompt H2D)");
  check_cuda(cudaMemcpy(gpu_embed_tokens, embed_cpu.data(), embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(embed H2D)");
  check_cuda(cudaMemcpy(gpu_layernorm_weights, layernorm_cpu.data(), layernorm_cpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(layernorm H2D)");
  check_cuda(cudaMemcpy(gpu_ffn_layernorm_weights, layernorm_cpu.data(), layernorm_cpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(ffn layernorm H2D)");

  // Build sparse deterministic projection weights in [out, in] row-major layout.
  w_q_cpu[0 * llama_prefill::HIDDEN_SIZE + 0] = __float2bfloat16(1.0f);
  w_q_cpu[1 * llama_prefill::HIDDEN_SIZE + 1] = __float2bfloat16(2.0f);
  w_q_cpu[2 * llama_prefill::HIDDEN_SIZE + 2] = __float2bfloat16(-0.5f);

  w_k_cpu[0 * llama_prefill::HIDDEN_SIZE + 0] = __float2bfloat16(1.5f);
  w_k_cpu[1 * llama_prefill::HIDDEN_SIZE + 1] = __float2bfloat16(-1.0f);
  w_k_cpu[2 * llama_prefill::HIDDEN_SIZE + 2] = __float2bfloat16(0.25f);

  w_v_cpu[0 * llama_prefill::HIDDEN_SIZE + 0] = __float2bfloat16(-2.0f);
  w_v_cpu[1 * llama_prefill::HIDDEN_SIZE + 1] = __float2bfloat16(0.5f);
  w_v_cpu[2 * llama_prefill::HIDDEN_SIZE + 2] = __float2bfloat16(1.25f);

  w_o_cpu[0 * llama_prefill::HIDDEN_SIZE + 0] = __float2bfloat16(1.0f);
  w_o_cpu[1 * llama_prefill::HIDDEN_SIZE + 1] = __float2bfloat16(1.0f);
  w_o_cpu[2 * llama_prefill::HIDDEN_SIZE + 2] = __float2bfloat16(1.0f);

  check_cuda(cudaMemcpy(gpu_w_q, w_q_cpu.data(), w_q_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w_q H2D)");
  check_cuda(cudaMemcpy(gpu_w_k, w_k_cpu.data(), w_k_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w_k H2D)");
  check_cuda(cudaMemcpy(gpu_w_v, w_v_cpu.data(), w_v_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w_v H2D)");
  check_cuda(cudaMemcpy(gpu_w_o, w_o_cpu.data(), w_o_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w_o H2D)");
  check_cuda(cudaMemcpy(gpu_w_gate, w_gate_cpu.data(), w_gate_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w_gate H2D)");
  check_cuda(cudaMemcpy(gpu_w_up, w_up_cpu.data(), w_up_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w_up H2D)");
  check_cuda(cudaMemcpy(gpu_w_down, w_down_cpu.data(), w_down_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w_down H2D)");

  std::vector<__nv_bfloat16> residual_input_cpu(prompt_len * llama_prefill::HIDDEN_SIZE, __float2bfloat16(0.25f));
  std::vector<__nv_bfloat16> residual_embed_cpu(prompt_len * llama_prefill::HIDDEN_SIZE, __float2bfloat16(0.75f));
  check_cuda(cudaMemcpy(gpu_residual_input, residual_input_cpu.data(), hidden_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(residual input H2D)");
  check_cuda(cudaMemcpy(gpu_residual_embeds, residual_embed_cpu.data(), hidden_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(residual embeds H2D)");

  llama_prefill::residual_add(gpu_residual_input, gpu_residual_embeds, prompt_len);

  std::vector<__nv_bfloat16> residual_out(prompt_len * llama_prefill::HIDDEN_SIZE);
  check_cuda(cudaMemcpy(residual_out.data(), gpu_residual_input, hidden_bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy(residual D2H)");

  const std::vector<size_t> residual_checks{
      0,
      1023,
      1024,
      llama_prefill::HIDDEN_SIZE,
      llama_prefill::HIDDEN_SIZE + 1024,
      prompt_len * llama_prefill::HIDDEN_SIZE - 1};
  for (const size_t idx : residual_checks) {
    require(std::fabs(__bfloat162float(residual_out[idx]) - 1.0f) < 0.01f, "residual_add produced unexpected value");
  }

  llama_prefill::PrefillWeights weights;
  weights.tok_embeddings = gpu_embed_tokens;
  weights.input_layernorm.push_back(gpu_layernorm_weights);
  weights.post_attention_layernorm.push_back(gpu_ffn_layernorm_weights);
  weights.w_q.push_back(gpu_w_q);
  weights.w_k.push_back(gpu_w_k);
  weights.w_v.push_back(gpu_w_v);
  weights.w_o.push_back(gpu_w_o);
  weights.w_gate.push_back(gpu_w_gate);
  weights.w_up.push_back(gpu_w_up);
  weights.w_down.push_back(gpu_w_down);
  weights.norm = gpu_layernorm_weights;
  weights.vocab_size = vocab_size;

  llama_prefill::PagedAttentionState paged_attention_state;
  paged_attention_state.slot = 0;
  paged_attention_state.block_size = 16;
  paged_attention_state.max_blocks_per_seq = llama_prefill::MAX_BLOCKS_PER_SEQ;

  const size_t per_kv_cache_bytes =
      static_cast<size_t>(paged_attention_state.block_size) * llama_prefill::KV_DIM * sizeof(__nv_bfloat16);
  paged_attention_state.v_offset = per_kv_cache_bytes;
  paged_attention_state.block_bytes = per_kv_cache_bytes * 2;
  check_cuda(cudaMalloc(&paged_attention_state.kv_cache, paged_attention_state.block_bytes),
             "cudaMalloc(paged_attention_state.kv_cache)");

  const size_t block_table_elems =
      static_cast<size_t>(llama_prefill::MAX_SEQUENCES) * llama_prefill::N_LAYERS * llama_prefill::MAX_BLOCKS_PER_SEQ;
  int* block_table = nullptr;
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&block_table), block_table_elems * sizeof(int), cudaHostAllocMapped),
             "cudaHostAlloc(block_table)");
  std::fill_n(block_table, block_table_elems, -1);
  std::vector<int> free_blocks_storage{0};
  paged_attention_state.block_table = block_table;
  paged_attention_state.free_blocks = free_blocks_storage.data();
  paged_attention_state.free_blocks_count = free_blocks_storage.size();

  std::vector<std::vector<int>> generated_tokens(1);
  std::vector<int> last_generated_tokens(1);
  std::vector<int> current_prompt_len(1);
  llama_prefill::prefill(gpu_prompt_tokens, prompt_len, gpu_input_embeddings, gpu_hidden_state, gpu_rms_norms, gpu_q_proj,
                         gpu_k_proj_batched_buffer, gpu_v_proj_batched_buffer, gpu_mlp_gate, gpu_mlp_up, weights,
                         &paged_attention_state, gpu_prefill_attn_scores, gpu_embed_proj, generated_tokens,
                         last_generated_tokens, current_prompt_len);

  std::vector<__nv_bfloat16> hidden_out(prompt_len * llama_prefill::HIDDEN_SIZE);
  check_cuda(cudaMemcpy(hidden_out.data(), gpu_hidden_state, hidden_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(hidden D2H)");

  const float inv_rms = 1.0f / std::sqrt(1.0f + 1.0e-5f);

  std::vector<__nv_bfloat16> k_proj_out(prompt_len * llama_prefill::KV_DIM);
  std::vector<__nv_bfloat16> v_proj_out(prompt_len * llama_prefill::KV_DIM);
  check_cuda(cudaMemcpy(k_proj_out.data(), gpu_k_proj_batched_buffer, kv_proj_bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy(k_proj D2H)");
  check_cuda(cudaMemcpy(v_proj_out.data(), gpu_v_proj_batched_buffer, kv_proj_bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy(v_proj D2H)");

  require(std::fabs(__bfloat162float(hidden_out[0])) < 10.0f, "hidden_state should contain finite o_proj output");
  for (size_t token = 0; token < prompt_len; ++token) {
    const size_t kv_base = token * llama_prefill::KV_DIM;

    const float token_pos = static_cast<float>(token);
    const float k_pair0_angle = token_pos * rope_theta_for_pair(0, 64);
    const float k_pair1_angle = token_pos * rope_theta_for_pair(1, 64);
    const float k_in0 = 1.5f * inv_rms;
    const float k_in1 = -1.0f * inv_rms;
    const float k_in2 = 0.25f * inv_rms;
    const float k_expected0 = k_in0 * std::cos(k_pair0_angle) - k_in1 * std::sin(k_pair0_angle);
    const float k_expected1 = k_in0 * std::sin(k_pair0_angle) + k_in1 * std::cos(k_pair0_angle);
    const float k_expected2 = k_in2 * std::cos(k_pair1_angle);
    const float k_expected3 = k_in2 * std::sin(k_pair1_angle);

    require(std::fabs(__bfloat162float(k_proj_out[kv_base + 0]) - k_expected0) < 0.05f, "k_proj dim0 mismatch");
    require(std::fabs(__bfloat162float(k_proj_out[kv_base + 1]) - k_expected1) < 0.05f, "k_proj dim1 mismatch");
    require(std::fabs(__bfloat162float(k_proj_out[kv_base + 2]) - k_expected2) < 0.03f, "k_proj dim2 mismatch");
    require(std::fabs(__bfloat162float(k_proj_out[kv_base + 3]) - k_expected3) < 0.03f, "k_proj dim3 mismatch");

    require(std::fabs(__bfloat162float(v_proj_out[kv_base + 0]) - (-2.0f * inv_rms)) < 0.04f, "v_proj dim0 mismatch");
    require(std::fabs(__bfloat162float(v_proj_out[kv_base + 1]) - (0.5f * inv_rms)) < 0.02f, "v_proj dim1 mismatch");
    require(std::fabs(__bfloat162float(v_proj_out[kv_base + 2]) - (1.25f * inv_rms)) < 0.03f, "v_proj dim2 mismatch");
    require(std::fabs(__bfloat162float(v_proj_out[kv_base + 3])) < 0.01f, "v_proj dim3 should stay near zero");
  }

  check_cuda(cudaFree(paged_attention_state.kv_cache), "cudaFree(paged_attention_state.kv_cache)");
  check_cuda(cudaFreeHost(block_table), "cudaFreeHost(block_table)");
  check_cuda(cudaFreeHost(gpu_embed_proj), "cudaFreeHost(gpu_embed_proj)");
  check_cuda(cudaFree(gpu_prefill_attn_scores), "cudaFree(gpu_prefill_attn_scores)");
  check_cuda(cudaFree(gpu_mlp_up), "cudaFree(gpu_mlp_up)");
  check_cuda(cudaFree(gpu_mlp_gate), "cudaFree(gpu_mlp_gate)");
  check_cuda(cudaFree(gpu_ffn_layernorm_weights), "cudaFree(gpu_ffn_layernorm_weights)");
  check_cuda(cudaFree(gpu_residual_embeds), "cudaFree(gpu_residual_embeds)");
  check_cuda(cudaFree(gpu_residual_input), "cudaFree(gpu_residual_input)");
  check_cuda(cudaFree(gpu_w_down), "cudaFree(gpu_w_down)");
  check_cuda(cudaFree(gpu_w_up), "cudaFree(gpu_w_up)");
  check_cuda(cudaFree(gpu_w_gate), "cudaFree(gpu_w_gate)");
  check_cuda(cudaFree(gpu_w_o), "cudaFree(gpu_w_o)");
  check_cuda(cudaFree(gpu_w_v), "cudaFree(gpu_w_v)");
  check_cuda(cudaFree(gpu_w_k), "cudaFree(gpu_w_k)");
  check_cuda(cudaFree(gpu_w_q), "cudaFree(gpu_w_q)");
  check_cuda(cudaFree(gpu_v_proj_batched_buffer), "cudaFree(gpu_v_proj_batched_buffer)");
  check_cuda(cudaFree(gpu_k_proj_batched_buffer), "cudaFree(gpu_k_proj_batched_buffer)");
  check_cuda(cudaFree(gpu_q_proj), "cudaFree(gpu_q_proj)");
  check_cuda(cudaFree(gpu_layernorm_weights), "cudaFree(gpu_layernorm_weights)");
  check_cuda(cudaFree(gpu_rms_norms), "cudaFree(gpu_rms_norms)");
  check_cuda(cudaFree(gpu_hidden_state), "cudaFree(gpu_hidden_state)");
  check_cuda(cudaFree(gpu_input_embeddings), "cudaFree(gpu_input_embeddings)");
  check_cuda(cudaFree(gpu_embed_tokens), "cudaFree(gpu_embed_tokens)");
  check_cuda(cudaFree(gpu_prompt_tokens), "cudaFree(gpu_prompt_tokens)");
  return 0;
}

}  // namespace

int main() {
  try {
    return run_prefill_test();
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
