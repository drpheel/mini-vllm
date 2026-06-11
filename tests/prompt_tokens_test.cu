#include "../src/prompt_tokens.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::filesystem::path temp_file_path(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("cuda_cublas_json_" + name + ".txt");
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path);
  require(static_cast<bool>(output), "Could not open temp file for writing: " + path.string());
  output << text;
}

void test_space_delimited_round_trip() {
  const std::vector<int> expected{791, 6864, 315, 9822, 374};
  const auto path = temp_file_path("prompt_tokens_space");
  write_text(path, "791 6864 315 9822 374\n");

  const prompt_tokens::GpuPromptTokens gpu_tokens = prompt_tokens::load_token_ids_to_gpu(path);
  require(gpu_tokens.count == expected.size(), "Unexpected GPU token count");
  require(gpu_tokens.bytes() == expected.size() * sizeof(int), "Unexpected GPU token byte count");

  std::vector<int> actual(gpu_tokens.count);
  check_cuda(cudaMemcpy(actual.data(), gpu_tokens.device_ptr, gpu_tokens.bytes(), cudaMemcpyDeviceToHost),
             "cudaMemcpy(prompt tokens D2H)");
  require(actual == expected, "GPU prompt token round trip did not preserve IDs");

  std::filesystem::remove(path);
}

void test_json_array_parse() {
  const auto path = temp_file_path("prompt_tokens_json");
  write_text(path, "[128000, 791, 6864, 315]\n");

  const std::vector<int> tokens = prompt_tokens::read_token_ids(path);
  require((tokens == std::vector<int>{128000, 791, 6864, 315}), "JSON-style tokenizer output was not parsed");

  std::filesystem::remove(path);
}

void test_max_prompt_len() {
  const auto path = temp_file_path("prompt_tokens_too_long");
  {
    std::ofstream output(path);
    require(static_cast<bool>(output), "Could not open temp file for writing: " + path.string());
    for (int i = 0; i <= prompt_tokens::MAX_PROMPT_LEN; ++i) {
      if (i != 0) {
        output << ' ';
      }
      output << i;
    }
    output << '\n';
  }

  bool rejected = false;
  try {
    (void)prompt_tokens::read_token_ids(path);
  } catch (const std::runtime_error&) {
    rejected = true;
  }

  std::filesystem::remove(path);
  require(rejected, "Prompt longer than MAX_PROMPT_LEN was accepted");
}

int run_all_tests() {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "Skipping CUDA prompt token test: no CUDA device available\n";
    return 77;
  }

  test_space_delimited_round_trip();
  test_json_array_parse();
  test_max_prompt_len();
  return 0;
}

}  // namespace

int main() {
  try {
    return run_all_tests();
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
