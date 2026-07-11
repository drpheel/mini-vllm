#pragma once

#include <cuda_runtime.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prompt_tokens {

constexpr int MAX_PROMPT_LEN = 512;

inline void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
  }
}

struct GpuPromptTokens {
  int* device_ptr = nullptr;
  size_t count = 0;

  GpuPromptTokens() = default;

  GpuPromptTokens(int* device_ptr_in, size_t count_in) : device_ptr(device_ptr_in), count(count_in) {}

  GpuPromptTokens(const GpuPromptTokens&) = delete;
  GpuPromptTokens& operator=(const GpuPromptTokens&) = delete;

  GpuPromptTokens(GpuPromptTokens&& other) noexcept : device_ptr(std::exchange(other.device_ptr, nullptr)), count(other.count) {
    other.count = 0;
  }

  GpuPromptTokens& operator=(GpuPromptTokens&& other) noexcept {
    if (this != &other) {
      reset();
      device_ptr = std::exchange(other.device_ptr, nullptr);
      count = other.count;
      other.count = 0;
    }
    return *this;
  }

  ~GpuPromptTokens() {
    reset();
  }

  size_t bytes() const {
    return count * sizeof(int);
  }

  void reset() noexcept {
    if (device_ptr != nullptr) {
      cudaFreeHost(device_ptr);
      device_ptr = nullptr;
    }
    count = 0;
  }
};

inline bool is_token_delimiter(char value) {
  const auto byte = static_cast<unsigned char>(value);
  return std::isspace(byte) || value == ',' || value == '[' || value == ']';
}

inline std::vector<int> read_token_ids(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open prompt token file: " + path.string());
  }

  const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  std::vector<int> tokens;
  size_t pos = 0;

  while (pos < text.size()) {
    while (pos < text.size() && is_token_delimiter(text[pos])) {
      ++pos;
    }
    if (pos >= text.size()) {
      break;
    }

    const size_t start = pos;
    if (text[pos] == '+' || text[pos] == '-') {
      ++pos;
    }
    const size_t digits_start = pos;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    if (digits_start == pos) {
      throw std::runtime_error("Invalid token ID text near byte " + std::to_string(start) + " in " + path.string());
    }
    if (pos < text.size() && !is_token_delimiter(text[pos])) {
      throw std::runtime_error("Invalid token ID delimiter near byte " + std::to_string(pos) + " in " + path.string());
    }

    const int64_t value = std::stoll(text.substr(start, pos - start));
    if (value < 0 || value > std::numeric_limits<int>::max()) {
      throw std::runtime_error("Token ID out of int range in " + path.string());
    }

    tokens.push_back(static_cast<int>(value));
    if (tokens.size() > MAX_PROMPT_LEN) {
      throw std::runtime_error("Prompt token count exceeds MAX_PROMPT_LEN=" + std::to_string(MAX_PROMPT_LEN));
    }
  }

  if (tokens.empty()) {
    throw std::runtime_error("Prompt token file is empty: " + path.string());
  }

  return tokens;
}

inline GpuPromptTokens copy_token_ids_to_gpu(const std::vector<int>& input_tokens) {
  if (input_tokens.empty()) {
    throw std::runtime_error("No prompt tokens to copy");
  }
  if (input_tokens.size() > MAX_PROMPT_LEN) {
    throw std::runtime_error("Prompt token count exceeds MAX_PROMPT_LEN=" + std::to_string(MAX_PROMPT_LEN));
  }

  // Jetson UMA: mapped host memory lets the host fill token IDs without a H2D memcpy.
  void* raw_device_tokens = nullptr;
  const size_t byte_count = input_tokens.size() * sizeof(int);
  check_cuda(cudaHostAlloc(&raw_device_tokens, byte_count, cudaHostAllocMapped),
             "cudaHostAlloc(prompt tokens)");
  void* device_alias = nullptr;
  check_cuda(cudaHostGetDevicePointer(&device_alias, raw_device_tokens, 0),
             "cudaHostGetDevicePointer(prompt tokens)");
  if (device_alias != raw_device_tokens) {
    cudaFreeHost(raw_device_tokens);
    throw std::runtime_error("prompt tokens: expected unified host/device pointer on Jetson");
  }

  auto* device_tokens = static_cast<int*>(raw_device_tokens);
  std::memcpy(device_tokens, input_tokens.data(), byte_count);

  return GpuPromptTokens(device_tokens, input_tokens.size());
}

inline GpuPromptTokens load_token_ids_to_gpu(const std::filesystem::path& path) {
  return copy_token_ids_to_gpu(read_token_ids(path));
}

}  // namespace prompt_tokens
