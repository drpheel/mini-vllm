#include "../src/prefill.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
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
  const size_t embed_bytes = static_cast<size_t>(vocab_size) * llama_prefill::HIDDEN_SIZE * sizeof(__nv_bfloat16);

  std::vector<__nv_bfloat16> embed_cpu(static_cast<size_t>(vocab_size) * llama_prefill::HIDDEN_SIZE, __float2bfloat16(1.0f));
  std::vector<__nv_bfloat16> layernorm_cpu(llama_prefill::HIDDEN_SIZE, __float2bfloat16(1.0f));

  int* gpu_prompt_tokens = nullptr;
  __nv_bfloat16* gpu_embed_tokens = nullptr;
  __nv_bfloat16* gpu_input_embeddings = nullptr;
  __nv_bfloat16* gpu_hidden_state = nullptr;
  __nv_bfloat16* gpu_rms_norms = nullptr;
  __nv_bfloat16* gpu_layernorm_weights = nullptr;

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_prompt_tokens), prompt_tokens.size() * sizeof(int)),
             "cudaMalloc(gpu_prompt_tokens)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_embed_tokens), embed_bytes), "cudaMalloc(gpu_embed_tokens)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_input_embeddings), hidden_bytes), "cudaMalloc(gpu_input_embeddings)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_hidden_state), hidden_bytes), "cudaMalloc(gpu_hidden_state)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_rms_norms), hidden_bytes), "cudaMalloc(gpu_rms_norms)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_layernorm_weights), layernorm_cpu.size() * sizeof(__nv_bfloat16)),
             "cudaMalloc(gpu_layernorm_weights)");

  check_cuda(cudaMemcpy(gpu_prompt_tokens, prompt_tokens.data(), prompt_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
             "cudaMemcpy(prompt H2D)");
  check_cuda(cudaMemcpy(gpu_embed_tokens, embed_cpu.data(), embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(embed H2D)");
  check_cuda(cudaMemcpy(gpu_layernorm_weights, layernorm_cpu.data(), layernorm_cpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(layernorm H2D)");

  llama_prefill::PrefillWeights weights;
  weights.tok_embeddings = gpu_embed_tokens;
  weights.input_layernorm.push_back(gpu_layernorm_weights);

  llama_prefill::prefill(gpu_prompt_tokens, prompt_len, gpu_input_embeddings, gpu_hidden_state, gpu_rms_norms, weights);

  std::vector<__nv_bfloat16> hidden_out(prompt_len * llama_prefill::HIDDEN_SIZE);
  check_cuda(cudaMemcpy(hidden_out.data(), gpu_hidden_state, hidden_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(hidden D2H)");

  const float expected = 1.0f / std::sqrt(1.0f + 1.0e-5f);
  for (size_t i = 0; i < hidden_out.size(); ++i) {
    const float actual = __bfloat162float(hidden_out[i]);
    require(std::fabs(actual - expected) < 0.015f, "prefill output does not match expected RMS norm value");
  }

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
