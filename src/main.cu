#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#include "bench.cuh"
#include "decode.cuh"
#include "prompt_tokens.cuh"
#include "prefill.cuh"

#include <nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <malloc.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace {

constexpr double B_TO_MB = 1024.0 * 1024.0;
constexpr double B_TO_GB = 1024.0 * 1024.0 * 1024.0;
constexpr size_t COPY_CHUNK_BYTES = 64 * 1024 * 1024;
constexpr int LLAMA_HIDDEN_SIZE = llama_prefill::HIDDEN_SIZE;
constexpr int END_OF_TEXT_TOKEN_ID = 128001;  // <|end_of_text|>
constexpr int EOM_ID_TOKEN_ID = 128008;       // <|eom_id|>
constexpr int EOT_ID_TOKEN_ID = 128009;       // <|eot_id|>

bool is_stop_token(int token_id) {
  return token_id == END_OF_TEXT_TOKEN_ID || token_id == EOM_ID_TOKEN_ID || token_id == EOT_ID_TOKEN_ID;
}

struct EvalItem {
  std::string id;
  std::vector<int> tokens;
  int max_new = 128;
};

std::vector<EvalItem> load_eval_jsonl(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open eval jsonl: " + path);
  }
  std::vector<EvalItem> items;
  std::string line;
  size_t line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    if (line.empty()) {
      continue;
    }
    const nlohmann::json row = nlohmann::json::parse(line);
    EvalItem item;
    item.id = row.at("id").get<std::string>();
    item.tokens = row.at("tokens").get<std::vector<int>>();
    if (row.contains("max_new")) {
      item.max_new = row.at("max_new").get<int>();
    }
    if (item.id.empty() || item.max_new < 1) {
      throw std::runtime_error("eval jsonl line " + std::to_string(line_no) + " needs an id and positive max_new");
    }
    items.push_back(std::move(item));
  }
  if (items.empty()) {
    throw std::runtime_error("eval jsonl has no items: " + path);
  }
  return items;
}

// Longest prompt that still fits, without keeping the token lists. The lists are
// loaded only after the weight and KV allocations, so they cannot split that hole.
size_t scan_eval_max_prompt(const std::string& path, size_t* item_count) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open eval jsonl: " + path);
  }
  size_t max_prompt = 1;
  size_t count = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const nlohmann::json row = nlohmann::json::parse(line);
    const size_t n = row.at("tokens").size();
    ++count;
    if (n > 0 && n < static_cast<size_t>(prompt_tokens::MAX_PROMPT_LEN)) {
      max_prompt = std::max(max_prompt, n);
    }
  }
  if (count == 0) {
    throw std::runtime_error("eval jsonl has no items: " + path);
  }
  *item_count = count;
  return max_prompt;
}
constexpr int MAX_SEQ_LEN = llama_prefill::BLOCK_SIZE * llama_prefill::MAX_BLOCKS_PER_SEQ;
static_assert(prompt_tokens::MAX_PROMPT_LEN == MAX_SEQ_LEN, "prompt cap and KV context diverged");
static_assert(decode::MAX_BLOCKS_PER_SEQ == llama_prefill::MAX_BLOCKS_PER_SEQ, "decode page stride diverged");
constexpr int MAX_NEW_TOKENS = 1024;
constexpr int BEST_OF_N = 4;
// Temperature multinomial sampling. TEMPERATURE <= 0 falls back to greedy argmax.
constexpr float TEMPERATURE = 0.7f;
// Nucleus sampling: keep smallest set of tokens with cumulative prob >= TOP_P.
// TOP_P >= 1 disables the cutoff (sample full softmax).
constexpr float TOP_P = 0.9f;
constexpr uint32_t SAMPLE_SEED = 42;
constexpr const char* DEFAULT_MODEL_PATH = "/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors";

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* call) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(call) + " failed with cublas status " + std::to_string(static_cast<int>(status)));
  }
}

// Softmax(logits / temperature), optional top-p nucleus filter, then multinomial sample.
// temperature <= 0 => greedy argmax. top_p >= 1 => no nucleus cutoff.
// If out_logprob != nullptr, writes log-prob of the chosen token under the sampling distribution.
int sample_token(const std::vector<float>& logits,
                 float temperature,
                 float top_p,
                 std::mt19937& rng,
                 double* out_logprob = nullptr) {
  if (logits.empty()) {
    throw std::runtime_error("sample_token requires non-empty logits");
  }

  int best_idx = 0;
  float best_logit = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_logit) {
      best_logit = logits[i];
      best_idx = static_cast<int>(i);
    }
  }
  if (temperature <= 0.0f) {
    if (out_logprob != nullptr) {
      *out_logprob = 0.0;
    }
    return best_idx;
  }

  // Numerically stable softmax in float64, shifted by max logit.
  double sum = 0.0;
  std::vector<double> probs(logits.size());
  const double inv_temp = 1.0 / static_cast<double>(temperature);
  for (size_t i = 0; i < logits.size(); ++i) {
    const double weight = std::exp((static_cast<double>(logits[i]) - static_cast<double>(best_logit)) * inv_temp);
    probs[i] = weight;
    sum += weight;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    if (out_logprob != nullptr) {
      *out_logprob = 0.0;
    }
    return best_idx;
  }
  for (double& p : probs) {
    p /= sum;
  }

  auto set_logprob = [&](size_t idx, double p) {
    if (out_logprob != nullptr) {
      *out_logprob = (p > 0.0 && std::isfinite(p)) ? std::log(p) : -1.0e9;
    }
  };

  // Nucleus (top-p): sort by prob descending, keep until cumulative >= top_p, renorm, sample.
  std::vector<size_t> order(probs.size());
  std::iota(order.begin(), order.end(), 0);
  if (top_p > 0.0f && top_p < 1.0f) {
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
    double cumulative = 0.0;
    size_t nucleus = 0;
    for (; nucleus < order.size(); ++nucleus) {
      cumulative += probs[order[nucleus]];
      if (cumulative >= static_cast<double>(top_p)) {
        ++nucleus;
        break;
      }
    }
    if (nucleus == 0) {
      nucleus = 1;
    }
    order.resize(nucleus);
    double nucleus_sum = 0.0;
    for (size_t idx : order) {
      nucleus_sum += probs[idx];
    }
    if (!(nucleus_sum > 0.0) || !std::isfinite(nucleus_sum)) {
      set_logprob(static_cast<size_t>(best_idx), probs[static_cast<size_t>(best_idx)]);
      return best_idx;
    }
    std::uniform_real_distribution<double> dist(0.0, nucleus_sum);
    double needle = dist(rng);
    for (size_t idx : order) {
      needle -= probs[idx];
      if (needle <= 0.0) {
        set_logprob(idx, probs[idx] / nucleus_sum);
        return static_cast<int>(idx);
      }
    }
    set_logprob(order.back(), probs[order.back()] / nucleus_sum);
    return static_cast<int>(order.back());
  }

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double needle = dist(rng);
  for (size_t i = 0; i < probs.size(); ++i) {
    needle -= probs[i];
    if (needle <= 0.0) {
      set_logprob(i, probs[i]);
      return static_cast<int>(i);
    }
  }
  set_logprob(probs.size() - 1, probs.back());
  return static_cast<int>(logits.size() - 1);
}

// Rank best-of-N candidates: avg decode logprob + light length prior.
double score_generation_candidate(const std::vector<int>& tokens, double sum_logprob, int decode_steps) {
  const int n = static_cast<int>(tokens.size());
  const double avg_lp = decode_steps > 0 ? (sum_logprob / static_cast<double>(decode_steps)) : -1.0e9;
  double score = avg_lp;
  if (n >= 4 && n <= 40) {
    score += 0.5;
  }
  if (n <= 2) {
    score -= 2.0;  // Discourage one/two-token shortcuts.
  }
  return score;
}

struct RankedToken {
  int id = 0;
  double logprob = 0.0;
};

// Pick up to `count` distinct high-probability first tokens from prefill logits.
// Uses temperature softmax + top-p nucleus, then takes the top-ranked nucleus tokens
// (not independent multinomial draws, which often collapse to the same id).
std::vector<RankedToken> select_diverse_first_tokens(const std::vector<float>& logits,
                                                    int count,
                                                    float temperature,
                                                    float top_p) {
  if (logits.empty() || count <= 0) {
    return {};
  }

  int best_idx = 0;
  float best_logit = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_logit) {
      best_logit = logits[i];
      best_idx = static_cast<int>(i);
    }
  }

  std::vector<double> probs(logits.size(), 0.0);
  if (temperature <= 0.0f) {
    probs[static_cast<size_t>(best_idx)] = 1.0;
  } else {
    double sum = 0.0;
    const double inv_temp = 1.0 / static_cast<double>(temperature);
    for (size_t i = 0; i < logits.size(); ++i) {
      const int token_id = static_cast<int>(i);
      if (is_stop_token(token_id)) {
        continue;  // Never start a best-of-N answer with an immediate stop token.
      }
      const double weight =
          std::exp((static_cast<double>(logits[i]) - static_cast<double>(best_logit)) * inv_temp);
      probs[i] = weight;
      sum += weight;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
      probs.assign(logits.size(), 0.0);
      probs[static_cast<size_t>(best_idx)] = 1.0;
    } else {
      for (double& p : probs) {
        p /= sum;
      }
    }
  }

  std::vector<size_t> order(probs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });

  size_t nucleus = order.size();
  if (top_p > 0.0f && top_p < 1.0f) {
    double cumulative = 0.0;
    nucleus = 0;
    for (; nucleus < order.size(); ++nucleus) {
      if (probs[order[nucleus]] <= 0.0) {
        break;
      }
      cumulative += probs[order[nucleus]];
      if (cumulative >= static_cast<double>(top_p)) {
        ++nucleus;
        break;
      }
    }
    if (nucleus == 0) {
      nucleus = 1;
    }
  }

  std::vector<RankedToken> chosen;
  chosen.reserve(static_cast<size_t>(count));
  for (size_t i = 0; i < nucleus && static_cast<int>(chosen.size()) < count; ++i) {
    const size_t idx = order[i];
    if (probs[idx] <= 0.0) {
      continue;
    }
    chosen.push_back(RankedToken{static_cast<int>(idx), std::log(probs[idx])});
  }
  // If the nucleus is smaller than count, keep filling from the next-highest probs.
  for (size_t i = nucleus; i < order.size() && static_cast<int>(chosen.size()) < count; ++i) {
    const size_t idx = order[i];
    if (probs[idx] <= 0.0) {
      continue;
    }
    chosen.push_back(RankedToken{static_cast<int>(idx), std::log(probs[idx])});
  }
  return chosen;
}

// Jetson UMA zero-copy: host-mapped pages are GPU-visible without H2D/D2H memcpy.
// Prefer this over cudaMallocManaged for CPU-mutated control buffers (more reliable under
// heavy device allocations). Free with cudaFreeHost.
void* alloc_mapped_host(size_t bytes, const char* what) {
  void* host_ptr = nullptr;
  check_cuda(cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped), what);
  void* device_ptr = nullptr;
  check_cuda(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0), "cudaHostGetDevicePointer");
  if (device_ptr != host_ptr) {
    // Discrete GPUs can return a different device mapping; this path assumes Jetson UMA.
    cudaFreeHost(host_ptr);
    throw std::runtime_error(std::string(what) + ": expected unified host/device pointer on Jetson");
  }
  return host_ptr;
}

struct CublasHandleGuard {
  cublasHandle_t handle = nullptr;

  ~CublasHandleGuard() {
    (void)cublasDestroy(handle);
  }
};

void print_gpu_status() {
  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
  std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
  std::cout << "SM count: " << prop.multiProcessorCount << "\n";
  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << '\n';

  size_t free_mem = 0;
  size_t total_mem = 0;
  check_cuda(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo");
  std::cout << "Free memory: " << free_mem / B_TO_GB << " GB, total memory: " << total_mem / B_TO_GB << " GB\n";
}

std::string format_bytes(size_t bytes) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double amount = static_cast<double>(bytes);
  int unit_index = 0;
  while (amount >= 1024.0 && unit_index < 4) {
    amount /= 1024.0;
    ++unit_index;
  }

  std::ostringstream output;
  output << std::fixed << std::setprecision(2) << amount << ' ' << units[unit_index];
  return output.str();
}

uint64_t read_u64_le(const unsigned char bytes[8]) {
  uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | bytes[i];
  }
  return value;
}

size_t dtype_size_bytes(const std::string& dtype) {
  if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E5M2" || dtype == "F8_E4M3") {
    return 1;
  }
  if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16") {
    return 2;
  }
  if (dtype == "I32" || dtype == "U32" || dtype == "F32") {
    return 4;
  }
  if (dtype == "I64" || dtype == "U64" || dtype == "F64") {
    return 8;
  }
  throw std::runtime_error("Unsupported safetensors dtype: " + dtype);
}

struct TensorInfo {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  size_t begin = 0;
  size_t end = 0;

  size_t byte_size() const {
    return end - begin;
  }
};

struct SafetensorsFile {
  std::string path;
  size_t header_len = 0;
  size_t payload_start = 0;
  size_t payload_size = 0;
  size_t file_size = 0;
  std::unordered_map<std::string, TensorInfo> tensors;
};

struct TensorView {
  std::string dtype;
  std::vector<int64_t> shape;
  size_t offset = 0;
  size_t byte_size = 0;
  void* device_ptr = nullptr;
};

struct ModelWeights {
  void* device_block = nullptr;
  size_t total_bytes = 0;
  std::unordered_map<std::string, TensorView> tensors;

  __nv_bfloat16* tok_embeddings = nullptr;
  __nv_bfloat16* final_norm = nullptr;
  __nv_bfloat16* lm_head = nullptr;

  std::vector<__nv_bfloat16*> w_q;
  std::vector<__nv_bfloat16*> w_k;
  std::vector<__nv_bfloat16*> w_v;
  std::vector<__nv_bfloat16*> w_o;
  std::vector<__nv_bfloat16*> w_gate;
  std::vector<__nv_bfloat16*> w_up;
  std::vector<__nv_bfloat16*> w_down;
  std::vector<__nv_bfloat16*> rms_attn;
  std::vector<__nv_bfloat16*> rms_ffn;
};

size_t checked_element_count(const std::vector<int64_t>& shape, const std::string& name) {
  size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("Negative dimension in tensor " + name);
    }
    const auto unsigned_dim = static_cast<size_t>(dim);
    if (unsigned_dim != 0 && count > std::numeric_limits<size_t>::max() / unsigned_dim) {
      throw std::runtime_error("Shape overflow in tensor " + name);
    }
    count *= unsigned_dim;
  }
  return count;
}

SafetensorsFile read_safetensors_metadata(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open safetensors file: " + path);
  }

  unsigned char header_len_bytes[8]{};
  file.read(reinterpret_cast<char*>(header_len_bytes), sizeof(header_len_bytes));
  if (file.gcount() != static_cast<std::streamsize>(sizeof(header_len_bytes))) {
    throw std::runtime_error("File is too small to contain a safetensors header length: " + path);
  }

  const uint64_t header_len_u64 = read_u64_le(header_len_bytes);
  if (header_len_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("Safetensors header is too large for this platform");
  }

  SafetensorsFile metadata;
  metadata.path = path;
  metadata.header_len = static_cast<size_t>(header_len_u64);
  metadata.payload_start = sizeof(header_len_bytes) + metadata.header_len;
  metadata.file_size = static_cast<size_t>(std::filesystem::file_size(path));

  std::string header(metadata.header_len, '\0');
  file.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (file.gcount() != static_cast<std::streamsize>(header.size())) {
    throw std::runtime_error("Safetensors header is truncated: " + path);
  }

  const json parsed = json::parse(header);
  for (auto it = parsed.begin(); it != parsed.end(); ++it) {
    if (it.key() == "__metadata__") {
      continue;
    }

    const json& value = it.value();
    TensorInfo tensor;
    tensor.name = it.key();
    tensor.dtype = value.at("dtype").get<std::string>();
    tensor.shape = value.at("shape").get<std::vector<int64_t>>();

    const auto offsets = value.at("data_offsets").get<std::vector<size_t>>();
    if (offsets.size() != 2 || offsets[1] < offsets[0]) {
      throw std::runtime_error("Invalid offsets for tensor " + tensor.name);
    }
    tensor.begin = offsets[0];
    tensor.end = offsets[1];

    const size_t expected_bytes = checked_element_count(tensor.shape, tensor.name) * dtype_size_bytes(tensor.dtype);
    if (expected_bytes != tensor.byte_size()) {
      throw std::runtime_error("Shape/dtype byte count does not match offsets for tensor " + tensor.name);
    }
    if (tensor.dtype != "BF16") {
      throw std::runtime_error("Expected BF16 tensor but found " + tensor.dtype + " for " + tensor.name);
    }

    metadata.payload_size = std::max(metadata.payload_size, tensor.end);
    metadata.tensors.emplace(tensor.name, std::move(tensor));
  }

  const size_t expected_file_size = metadata.payload_start + metadata.payload_size;
  if (expected_file_size != metadata.file_size) {
    std::cout << "warning: header implies file size " << format_bytes(expected_file_size) << ", actual file size is "
              << format_bytes(metadata.file_size) << '\n';
  }

  return metadata;
}

void copy_payload_to_gpu(const SafetensorsFile& metadata, void* device_block) {
  std::ifstream file(metadata.path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not reopen safetensors file: " + metadata.path);
  }
  file.seekg(static_cast<std::streamoff>(metadata.payload_start), std::ios::beg);
  if (!file) {
    throw std::runtime_error("Could not seek to safetensors payload");
  }

  std::vector<char> chunk(COPY_CHUNK_BYTES);
  size_t copied = 0;
  auto* device_bytes = static_cast<char*>(device_block);

  while (copied < metadata.payload_size) {
    const size_t bytes_this_round = std::min(chunk.size(), metadata.payload_size - copied);
    file.read(chunk.data(), static_cast<std::streamsize>(bytes_this_round));
    if (file.gcount() != static_cast<std::streamsize>(bytes_this_round)) {
      throw std::runtime_error("Unexpected EOF while reading safetensors payload");
    }

    check_cuda(cudaMemcpy(device_bytes + copied, chunk.data(), bytes_this_round, cudaMemcpyHostToDevice),
               "cudaMemcpy(model payload chunk)");
    copied += bytes_this_round;

    std::cout << "Copied " << format_bytes(copied) << " / " << format_bytes(metadata.payload_size) << " to GPU\r"
              << std::flush;
  }
  std::cout << "\nFinished copying model payload to GPU\n";
}

bool parse_layer_tensor_name(const std::string& name, int* layer, std::string* suffix) {
  constexpr const char* prefix = "model.layers.";
  constexpr size_t prefix_len = 13;
  if (name.rfind(prefix, 0) != 0) {
    return false;
  }

  size_t pos = prefix_len;
  size_t value = 0;
  if (pos >= name.size() || !std::isdigit(static_cast<unsigned char>(name[pos]))) {
    return false;
  }
  while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
    value = value * 10 + static_cast<size_t>(name[pos] - '0');
    ++pos;
  }
  if (pos >= name.size() || name[pos] != '.') {
    return false;
  }

  *layer = static_cast<int>(value);
  *suffix = name.substr(pos + 1);
  return true;
}

__nv_bfloat16* bf16_tensor(ModelWeights& weights, const std::string& name, bool required = true) {
  const auto it = weights.tensors.find(name);
  if (it == weights.tensors.end()) {
    if (required) {
      throw std::runtime_error("Missing tensor: " + name);
    }
    return nullptr;
  }
  if (it->second.dtype != "BF16") {
    throw std::runtime_error("Expected BF16 tensor view for " + name);
  }
  return reinterpret_cast<__nv_bfloat16*>(it->second.device_ptr);
}

ModelWeights build_llama_weight_views(const SafetensorsFile& metadata, void* device_block) {
  ModelWeights weights;
  weights.device_block = device_block;
  weights.total_bytes = metadata.payload_size;

  auto* base = static_cast<char*>(device_block);
  for (const auto& [name, tensor] : metadata.tensors) {
    weights.tensors.emplace(name, TensorView{tensor.dtype, tensor.shape, tensor.begin, tensor.byte_size(), base + tensor.begin});
  }

  int max_layer = -1;
  for (const auto& [name, tensor] : metadata.tensors) {
    int layer = -1;
    std::string suffix;
    if (parse_layer_tensor_name(name, &layer, &suffix)) {
      max_layer = std::max(max_layer, layer);
    }
  }
  if (max_layer < 0) {
    throw std::runtime_error("No model.layers.N tensors found");
  }

  const int layer_count = max_layer + 1;
  weights.w_q.resize(layer_count);
  weights.w_k.resize(layer_count);
  weights.w_v.resize(layer_count);
  weights.w_o.resize(layer_count);
  weights.w_gate.resize(layer_count);
  weights.w_up.resize(layer_count);
  weights.w_down.resize(layer_count);
  weights.rms_attn.resize(layer_count);
  weights.rms_ffn.resize(layer_count);

  weights.tok_embeddings = bf16_tensor(weights, "model.embed_tokens.weight");
  weights.final_norm = bf16_tensor(weights, "model.norm.weight");
  weights.lm_head = bf16_tensor(weights, "lm_head.weight", false);

  for (int layer = 0; layer < layer_count; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    weights.w_q[layer] = bf16_tensor(weights, prefix + "self_attn.q_proj.weight");
    weights.w_k[layer] = bf16_tensor(weights, prefix + "self_attn.k_proj.weight");
    weights.w_v[layer] = bf16_tensor(weights, prefix + "self_attn.v_proj.weight");
    weights.w_o[layer] = bf16_tensor(weights, prefix + "self_attn.o_proj.weight");
    weights.w_gate[layer] = bf16_tensor(weights, prefix + "mlp.gate_proj.weight");
    weights.w_up[layer] = bf16_tensor(weights, prefix + "mlp.up_proj.weight");
    weights.w_down[layer] = bf16_tensor(weights, prefix + "mlp.down_proj.weight");
    weights.rms_attn[layer] = bf16_tensor(weights, prefix + "input_layernorm.weight");
    weights.rms_ffn[layer] = bf16_tensor(weights, prefix + "post_attention_layernorm.weight");
  }

  return weights;
}

void print_tensor_view(const ModelWeights& weights, const std::string& name) {
  const auto it = weights.tensors.find(name);
  if (it == weights.tensors.end()) {
    std::cout << name << " -> missing\n";
    return;
  }

  std::cout << name << " -> ptr=" << it->second.device_ptr << ", offset=" << it->second.offset
            << ", bytes=" << format_bytes(it->second.byte_size) << ", shape=[";
  for (size_t i = 0; i < it->second.shape.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << it->second.shape[i];
  }
  std::cout << "]\n";
}

void print_mapping_debug(const ModelWeights& weights) {
  std::cout << "\n== Weight pointer map ==\n";
  std::cout << "device block: " << weights.device_block << '\n';
  std::cout << "model bytes: " << weights.total_bytes << " (" << format_bytes(weights.total_bytes) << ")\n";
  std::cout << "layers: " << weights.w_k.size() << '\n';
  std::cout << "lm_head.weight: " << (weights.lm_head == nullptr ? "not present" : "present") << '\n';

  print_tensor_view(weights, "model.embed_tokens.weight");
  print_tensor_view(weights, "model.norm.weight");
  print_tensor_view(weights, "model.layers.0.self_attn.k_proj.weight");
  print_tensor_view(weights, "model.layers.5.self_attn.k_proj.weight");
  print_tensor_view(weights, "model.layers.15.self_attn.k_proj.weight");

  std::cout << "\nDirect pointer check: weights.w_k[5]=" << static_cast<void*>(weights.w_k.at(5)) << '\n';
}

void print_bf16_sample(const std::vector<__nv_bfloat16>& values) {
  std::cout << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < values.size(); ++i) {
    uint16_t bits = 0;
    std::memcpy(&bits, &values[i], sizeof(bits));
    std::cout << (i == 0 ? "" : ", ") << __bfloat162float(values[i]) << " (0x" << std::hex << std::setw(4)
              << std::setfill('0') << bits << std::dec << std::setfill(' ') << ")";
  }
  std::cout << '\n';
}

void print_input_embedding_debug(const prompt_tokens::GpuPromptTokens& gpu_input_tokens,
                                 __nv_bfloat16* input_embeddings) {
  if (gpu_input_tokens.count == 0) {
    return;
  }

  constexpr int SAMPLE_DIMS = 8;
  const size_t sample_tokens = std::min<size_t>(gpu_input_tokens.count, 3);
  std::cout << "\n== Gathered embedding sample ==\n";
  std::cout << "showing " << SAMPLE_DIMS << " values from dims 0.." << (SAMPLE_DIMS - 1) << " and 1024.."
            << (1024 + SAMPLE_DIMS - 1) << " for " << sample_tokens << " token(s)\n";

  for (size_t token_index = 0; token_index < sample_tokens; ++token_index) {
    std::vector<__nv_bfloat16> first_half(SAMPLE_DIMS);
    std::vector<__nv_bfloat16> second_half(SAMPLE_DIMS);
    const size_t row_offset = token_index * LLAMA_HIDDEN_SIZE;

    check_cuda(cudaMemcpy(first_half.data(), input_embeddings + row_offset, first_half.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(input embedding first half sample)");
    check_cuda(cudaMemcpy(second_half.data(), input_embeddings + row_offset + 1024,
                          second_half.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
               "cudaMemcpy(input embedding second half sample)");

    std::cout << "token embedding row " << token_index << " dims 0.." << (SAMPLE_DIMS - 1) << ": ";
    print_bf16_sample(first_half);
    std::cout << "token embedding row " << token_index << " dims 1024.." << (1024 + SAMPLE_DIMS - 1) << ": ";
    print_bf16_sample(second_half);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path = DEFAULT_MODEL_PATH;
    std::string prompt_tokens_path;
    bool bench_mode = false;
    bool bench_eager = false;
    bool bench_async = false;
    bool bench_graph = false;
    bool bench_warmup = true;
    int bench_steps = 32;
    std::string eval_jsonl_path;
    std::string eval_out_path;
    size_t eval_max_prompt_arg = 0;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--bench") {
        bench_mode = true;
        bench_eager = true;
        bench_async = true;
        bench_graph = true;
      } else if (arg == "--bench-eager") {
        bench_mode = true;
        bench_eager = true;
      } else if (arg == "--bench-async") {
        bench_mode = true;
        bench_async = true;
      } else if (arg == "--bench-graph") {
        bench_mode = true;
        bench_graph = true;
      } else if (arg == "--no-warmup") {
        bench_warmup = false;
      } else if (arg == "--eval-jsonl") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--eval-jsonl requires a path");
        }
        eval_jsonl_path = argv[++i];
      } else if (arg == "--eval-out") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--eval-out requires a path");
        }
        eval_out_path = argv[++i];
      } else if (arg == "--eval-max-prompt") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--eval-max-prompt requires a token count");
        }
        eval_max_prompt_arg = static_cast<size_t>(std::stoul(argv[++i]));
      } else if (arg == "--steps") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--steps requires a positive integer");
        }
        bench_steps = std::stoi(argv[++i]);
        if (bench_steps <= 0) {
          throw std::runtime_error("--steps must be positive");
        }
      } else if (arg == "--help" || arg == "-h") {
        std::cout << "usage: cuda_cublas_json [model.safetensors] [prompt_tokens.txt]\n"
                     "       [--bench | --bench-eager | --bench-async | --bench-graph] [--steps N] [--no-warmup]\n"
                     "       [--eval-jsonl requests.jsonl --eval-out results.jsonl]\n";
        return 0;
      } else if (!arg.empty() && arg[0] == '-') {
        throw std::runtime_error("unknown argument: " + arg);
      } else {
        positional.push_back(arg);
      }
    }
    if (positional.size() > 2) {
      throw std::runtime_error(
          "usage: cuda_cublas_json [model.safetensors] [prompt_tokens.txt] [--bench] [--steps N]");
    }
    if (!positional.empty()) {
      model_path = positional[0];
    }
    if (positional.size() > 1) {
      prompt_tokens_path = positional[1];
    }
    if (bench_mode && prompt_tokens_path.empty()) {
      throw std::runtime_error("benchmark mode requires a prompt token file");
    }
    const bool eval_mode = !eval_jsonl_path.empty() || !eval_out_path.empty();
    if (eval_mode && (eval_jsonl_path.empty() || eval_out_path.empty())) {
      throw std::runtime_error("--eval-jsonl and --eval-out are both required");
    }
    if (eval_mode && bench_mode) {
      throw std::runtime_error("eval mode cannot be combined with benchmark mode");
    }
    if (eval_mode && !prompt_tokens_path.empty()) {
      throw std::runtime_error("eval mode does not take a prompt token file");
    }
    std::vector<EvalItem> eval_items;
    size_t eval_item_count = 0;
    size_t eval_max_prompt = 1;
    if (eval_mode) {
      if (eval_max_prompt_arg > 0) {
        eval_max_prompt = eval_max_prompt_arg;
      } else {
        eval_max_prompt = scan_eval_max_prompt(eval_jsonl_path, &eval_item_count);
        malloc_trim(0);
      }
    }
    const int run_best_of_n = (bench_mode || eval_mode) ? 1 : BEST_OF_N;
    const float run_temperature = (bench_mode || eval_mode) ? 0.0f : TEMPERATURE;
    const float run_top_p = (bench_mode || eval_mode) ? 1.0f : TOP_P;

    print_gpu_status();
    prompt_tokens::GpuPromptTokens gpu_input_tokens;
    if (!eval_mode && !prompt_tokens_path.empty()) {
      gpu_input_tokens = prompt_tokens::load_token_ids_to_gpu(prompt_tokens_path);
      std::cout << "Loaded " << gpu_input_tokens.count << " prompt tokens to GPU at "
                << static_cast<void*>(gpu_input_tokens.device_ptr) << '\n';
    }

    // Decode writes the new token at row 0, so these buffers only need the prompt.
    // Allocate them after the weight block and the KV cache. A small buffer first
    // splits the only hole large enough for the 2.3 GB checkpoint.
    const size_t activation_tokens =
        std::max(eval_mode ? eval_max_prompt : gpu_input_tokens.count, static_cast<size_t>(1));
    __nv_bfloat16* input_embeddings = nullptr;
    const size_t input_embeddings_bytes = activation_tokens * LLAMA_HIDDEN_SIZE * sizeof(__nv_bfloat16);

    std::cout << "\nLoading safetensors file: " << model_path << '\n';

    const SafetensorsFile metadata = read_safetensors_metadata(model_path);
    std::cout << "Header bytes: " << metadata.header_len << " (" << format_bytes(metadata.header_len) << ")\n";
    std::cout << "Payload start: " << metadata.payload_start << '\n';
    std::cout << "Payload bytes: " << metadata.payload_size << " (" << format_bytes(metadata.payload_size) << ")\n";
    std::cout << "Tensor count: " << metadata.tensors.size() << '\n';

    int max_layer = -1;
    for (const auto& [name, tensor] : metadata.tensors) {
      (void)tensor;
      int layer = -1;
      std::string suffix;
      if (parse_layer_tensor_name(name, &layer, &suffix)) {
        max_layer = std::max(max_layer, layer);
      }
    }
    if (max_layer < 0) {
      throw std::runtime_error("No model.layers.N tensors found");
    }
    const size_t num_layers = static_cast<size_t>(max_layer + 1);

    void* model_weights = nullptr;
    check_cuda(cudaMalloc(&model_weights, metadata.payload_size), "cudaMalloc(model_weights)");
    std::cout << "Allocated one GPU block at " << model_weights << '\n';

    copy_payload_to_gpu(metadata, model_weights);
    ModelWeights weights = build_llama_weight_views(metadata, model_weights);

    // Cache immediately after the weight buffer, before the smaller activation
    // allocs split the leftover hole.
    llama_prefill::PagedAttentionState paged_attention_state;
    paged_attention_state.slot = 0;
    paged_attention_state.block_size = llama_prefill::BLOCK_SIZE;
    paged_attention_state.max_blocks_per_seq = llama_prefill::MAX_BLOCKS_PER_SEQ;
    const int total_blocks = std::max(1, static_cast<int>(num_layers) * paged_attention_state.max_blocks_per_seq);
    const size_t per_kv_cache_bytes =
        static_cast<size_t>(paged_attention_state.block_size) * llama_prefill::KV_DIM * sizeof(__nv_bfloat16);
    paged_attention_state.v_offset = per_kv_cache_bytes;
    paged_attention_state.block_bytes = per_kv_cache_bytes * 2;
    const size_t kv_cache_bytes = static_cast<size_t>(total_blocks) * paged_attention_state.block_bytes;
    check_cuda(cudaMalloc(&paged_attention_state.kv_cache, kv_cache_bytes),
               "cudaMalloc(paged_attention_state.kv_cache)");
    // No second copy of the cache. Best-of-N and the bench replays rerun prefill.
    // A full snapshot is another 128 MB, and this board does not have it after the weights.
    void* kv_snapshot = nullptr;
    std::cout << "Allocated KV cache " << format_bytes(kv_cache_bytes) << " for " << MAX_SEQ_LEN << " tokens\n";

    // Reserve the cuBLAS handle before the prompt activations fill the leftover hole.
    CublasHandleGuard cublas_guard;
    check_cublas(cublasCreate(&cublas_guard.handle), "cublasCreate(decode)");

    if (eval_mode) {
      gpu_input_tokens = prompt_tokens::allocate_token_capacity(eval_max_prompt);
      std::cout << "Eval token buffer " << eval_max_prompt;
      if (eval_item_count > 0) {
        std::cout << ", items " << eval_item_count;
      }
      std::cout << "\n";
    }

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&input_embeddings), input_embeddings_bytes),
               "cudaMalloc(input_embeddings)");
    std::cout << "Allocated prompt input embeddings at " << static_cast<void*>(input_embeddings) << " ("
              << format_bytes(input_embeddings_bytes) << ")\n";

    const size_t block_table_elems = static_cast<size_t>(llama_prefill::MAX_SEQUENCES) *
                                     llama_prefill::N_LAYERS * llama_prefill::MAX_BLOCKS_PER_SEQ;
    int* block_table =
        static_cast<int*>(alloc_mapped_host(block_table_elems * sizeof(int), "cudaHostAlloc(block_table)"));
    std::fill_n(block_table, block_table_elems, -1);
    std::vector<int> free_blocks_storage(static_cast<size_t>(total_blocks));
    std::iota(free_blocks_storage.begin(), free_blocks_storage.end(), 0);
    paged_attention_state.block_table = block_table;
    paged_attention_state.free_blocks = free_blocks_storage.data();
    paged_attention_state.free_blocks_count = free_blocks_storage.size();
    std::vector<std::vector<int>> generated_tokens(llama_prefill::MAX_SEQUENCES);
    std::vector<int> last_generated_tokens(llama_prefill::MAX_SEQUENCES);
    std::vector<int> current_prompt_len(llama_prefill::MAX_SEQUENCES);
    std::vector<char> is_slot_free(llama_prefill::MAX_SEQUENCES, 1);

    __nv_bfloat16* hidden_state = nullptr;
    __nv_bfloat16* rms_norms = nullptr;
    __nv_bfloat16* q_proj = nullptr;
    __nv_bfloat16* k_proj_batched_buffer = nullptr;
    __nv_bfloat16* v_proj_batched_buffer = nullptr;
    __nv_bfloat16* mlp_gate = nullptr;
    __nv_bfloat16* mlp_up = nullptr;
    __nv_bfloat16* prefill_attn_scores = nullptr;
    __nv_bfloat16* embed_proj = nullptr;
    const size_t kv_proj_bytes = activation_tokens * llama_prefill::KV_DIM * sizeof(__nv_bfloat16);
    const size_t mlp_intermediate_bytes =
        activation_tokens * llama_prefill::MLP_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&hidden_state), input_embeddings_bytes), "cudaMalloc(hidden_state)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&rms_norms), input_embeddings_bytes), "cudaMalloc(rms_norms)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&q_proj), input_embeddings_bytes), "cudaMalloc(q_proj)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&k_proj_batched_buffer), kv_proj_bytes),
               "cudaMalloc(k_proj_batched_buffer)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&v_proj_batched_buffer), kv_proj_bytes),
               "cudaMalloc(v_proj_batched_buffer)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&mlp_gate), mlp_intermediate_bytes), "cudaMalloc(mlp_gate)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&mlp_up), mlp_intermediate_bytes), "cudaMalloc(mlp_up)");

    llama_prefill::PrefillWeights prefill_weights;
    prefill_weights.tok_embeddings = weights.tok_embeddings;
    prefill_weights.input_layernorm = weights.rms_attn;
    prefill_weights.w_q = weights.w_q;
    prefill_weights.w_k = weights.w_k;
    prefill_weights.w_v = weights.w_v;
    prefill_weights.w_o = weights.w_o;
    prefill_weights.post_attention_layernorm = weights.rms_ffn;
    prefill_weights.w_gate = weights.w_gate;
    prefill_weights.w_up = weights.w_up;
    prefill_weights.w_down = weights.w_down;
    prefill_weights.norm = weights.final_norm;
    const auto embed_it = weights.tensors.find("model.embed_tokens.weight");
    if (embed_it == weights.tensors.end() || embed_it->second.shape.empty()) {
      throw std::runtime_error("model.embed_tokens.weight shape unavailable for vocab_size");
    }
    prefill_weights.vocab_size = static_cast<int>(embed_it->second.shape[0]);

    const size_t prefill_scores_bytes =
        activation_tokens * activation_tokens * sizeof(__nv_bfloat16);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&prefill_attn_scores), prefill_scores_bytes),
               "cudaMalloc(prefill_attn_scores)");
    const size_t embed_proj_bytes = static_cast<size_t>(prefill_weights.vocab_size) * sizeof(__nv_bfloat16);
    // Mapped host memory: GPU writes logits, CPU samples in place (no vocab-sized D2H).
    embed_proj = static_cast<__nv_bfloat16*>(alloc_mapped_host(embed_proj_bytes, "cudaHostAlloc(embed_proj)"));

    if (bench_mode && (gpu_input_tokens.count == 0 ||
                       gpu_input_tokens.count + static_cast<size_t>(bench_steps) >= static_cast<size_t>(MAX_SEQ_LEN))) {
      throw std::runtime_error("benchmark prompt plus --steps must fit in MAX_SEQ_LEN (" + std::to_string(MAX_SEQ_LEN) +
                               ")");
    }
    auto reset_kv_allocator = [&]() {
      std::fill_n(block_table, block_table_elems, -1);
      std::iota(free_blocks_storage.begin(), free_blocks_storage.end(), 0);
      paged_attention_state.free_blocks = free_blocks_storage.data();
      paged_attention_state.free_blocks_count = free_blocks_storage.size();
      generated_tokens[0].clear();
    };
    auto run_prefill = [&]() {
      llama_prefill::prefill(gpu_input_tokens.device_ptr, gpu_input_tokens.count, input_embeddings, hidden_state, rms_norms,
                             q_proj, k_proj_batched_buffer, v_proj_batched_buffer, mlp_gate, mlp_up, prefill_weights,
                             &paged_attention_state, prefill_attn_scores, embed_proj, generated_tokens,
                             last_generated_tokens, current_prompt_len);
    };
    if (bench_mode && bench_warmup) {
      nvtxRangePushA("prefill_warmup");
      run_prefill();
      nvtxRangePop();
      reset_kv_allocator();
    }
    float prefill_ms = 0.0f;
    if (bench_mode) {
      cudaEvent_t prefill_start = nullptr;
      cudaEvent_t prefill_stop = nullptr;
      check_cuda(cudaEventCreate(&prefill_start), "cudaEventCreate(prefill_start)");
      check_cuda(cudaEventCreate(&prefill_stop), "cudaEventCreate(prefill_stop)");
      nvtxRangePushA("prefill");
      check_cuda(cudaEventRecord(prefill_start), "cudaEventRecord(prefill_start)");
      run_prefill();
      check_cuda(cudaEventRecord(prefill_stop), "cudaEventRecord(prefill_stop)");
      check_cuda(cudaEventSynchronize(prefill_stop), "cudaEventSynchronize(prefill)");
      nvtxRangePop();
      check_cuda(cudaEventElapsedTime(&prefill_ms, prefill_start, prefill_stop), "cudaEventElapsedTime(prefill)");
      check_cuda(cudaEventDestroy(prefill_start), "cudaEventDestroy(prefill_start)");
      check_cuda(cudaEventDestroy(prefill_stop), "cudaEventDestroy(prefill_stop)");
    } else if (!eval_mode) {
      run_prefill();
    }
    if (!eval_mode) {
      std::cout << "Gathered " << gpu_input_tokens.count << " token embeddings into "
                << static_cast<void*>(input_embeddings) << '\n';
    }
    const int slot = paged_attention_state.slot;
    is_slot_free[static_cast<size_t>(slot)] = 0;
    const int embedding_length = LLAMA_HIDDEN_SIZE;
    const int mlp_intermediate_size = llama_prefill::MLP_INTERMEDIATE_SIZE;
    const int num_decode_tokens = 1;
    const int num_active_slots = 1;
    const float q_proj_alpha = 1.0f;
    const float q_proj_beta = 0.0f;
    const float k_proj_alpha = 1.0f;
    const float k_proj_beta = 0.0f;
    const float v_proj_alpha = 1.0f;
    const float v_proj_beta = 0.0f;
    const float o_proj_alpha = 1.0f;
    const float o_proj_beta = 0.0f;
    const float gate_alpha = 1.0f;
    const float gate_beta = 0.0f;
    const float up_alpha = 1.0f;
    const float up_beta = 0.0f;
    const float down_alpha = 1.0f;
    const float down_beta = 0.0f;
    const float embed_alpha = 1.0f;
    const float embed_beta = 0.0f;
    const int vocab_size = prefill_weights.vocab_size;
    const int kv_dim = llama_prefill::KV_DIM;

    int* decode_token_gpu =
        static_cast<int*>(alloc_mapped_host(sizeof(int), "cudaHostAlloc(decode_token_gpu)"));

    if (eval_mode) {
      eval_items = load_eval_jsonl(eval_jsonl_path);
    }

    int* gpu_seq_lens = static_cast<int*>(alloc_mapped_host(sizeof(int), "cudaHostAlloc(gpu_seq_lens)"));
    int* gpu_active_slots = static_cast<int*>(alloc_mapped_host(sizeof(int), "cudaHostAlloc(gpu_active_slots)"));
    *gpu_active_slots = slot;

    bool verified_first_decode_embed = false;
    const bool verbose_decode = (BEST_OF_N == 1);
    std::cout << "Decode sampling: temperature=" << run_temperature << " top_p=" << run_top_p
              << " max_new_tokens=" << (bench_mode ? bench_steps : MAX_NEW_TOKENS)
              << " max_seq_len=" << MAX_SEQ_LEN << " best_of_n=" << run_best_of_n
              << " diverse_first_tokens=1 seed=" << SAMPLE_SEED;
    if (eval_mode) {
      std::cout << " eval_items=" << eval_items.size() << " greedy=1";
    }
    std::cout << '\n';

    std::ofstream eval_out;
    if (eval_mode) {
      eval_out.open(eval_out_path, std::ios::app);
      if (!eval_out) {
        throw std::runtime_error("could not open eval output: " + eval_out_path);
      }
    }
    const size_t eval_passes = eval_mode ? eval_items.size() : 1;
    const size_t token_capacity = gpu_input_tokens.count;
    bool bench_tokens_match = true;
    for (size_t eval_pass = 0; eval_pass < eval_passes; ++eval_pass) {
      int run_max_new = bench_mode ? bench_steps : MAX_NEW_TOKENS;
      const auto eval_t0 = std::chrono::steady_clock::now();
      if (eval_mode) {
        const EvalItem& item = eval_items[eval_pass];
        if (item.tokens.empty() || item.tokens.size() > token_capacity ||
            item.tokens.size() >= static_cast<size_t>(MAX_SEQ_LEN)) {
          nlohmann::json row;
          row["id"] = item.id;
          row["error"] = "prompt does not fit in context";
          row["prompt_tokens"] = item.tokens.size();
          eval_out << row.dump() << '\n';
          eval_out.flush();
          continue;
        }
        if (item.tokens.size() > token_capacity) {
          throw std::runtime_error("eval prompt exceeds the allocated token buffer");
        }
        std::memcpy(gpu_input_tokens.device_ptr, item.tokens.data(), item.tokens.size() * sizeof(int));
        gpu_input_tokens.count = item.tokens.size();
        reset_kv_allocator();
        run_prefill();
        const int room = MAX_SEQ_LEN - static_cast<int>(item.tokens.size());
        run_max_new = std::min(item.max_new, std::min(room, MAX_NEW_TOKENS));
        if (run_max_new < 1) {
          nlohmann::json row;
          row["id"] = item.id;
          row["error"] = "no room to generate";
          row["prompt_tokens"] = item.tokens.size();
          eval_out << row.dump() << '\n';
          eval_out.flush();
          continue;
        }
      }

    // Prefill KV covers the prompt only; the greedy first token is not in cache yet.
    // Capture last-position logits and choose distinct high-prob first tokens for best-of-N.
    const int prompt_len_snapshot = current_prompt_len[static_cast<size_t>(slot)];
    // Prefill writes one logit row, for the last prompt token.
    const size_t first_logit_offset = 0;
    std::vector<float> first_token_logits(static_cast<size_t>(vocab_size));
    for (int token_idx = 0; token_idx < vocab_size; ++token_idx) {
      first_token_logits[static_cast<size_t>(token_idx)] =
          __bfloat162float(embed_proj[first_logit_offset + static_cast<size_t>(token_idx)]);
    }
    const std::vector<RankedToken> first_token_choices =
        select_diverse_first_tokens(first_token_logits, run_best_of_n, run_temperature, run_top_p);
    if (first_token_choices.empty()) {
      throw std::runtime_error("diverse first-token selection returned no candidates");
    }
    std::cout << "Prefill greedy first token was " << last_generated_tokens[static_cast<size_t>(slot)]
              << "; diverse first tokens:";
    for (const RankedToken& tok : first_token_choices) {
      std::cout << ' ' << tok.id;
    }
    std::cout << '\n';

    std::vector<int> block_table_snapshot(block_table, block_table + block_table_elems);
    std::vector<int> free_blocks_snapshot = free_blocks_storage;
    const size_t free_blocks_count_snapshot = paged_attention_state.free_blocks_count;

    std::vector<int> best_tokens;
    double best_score = -std::numeric_limits<double>::infinity();
    int best_candidate = -1;
    double bench_gpu_ms = 0.0;
    double bench_sample_ms = 0.0;
    int bench_forwards = 0;
    std::vector<int> bench_eager_outputs;
    cudaEvent_t bench_step_start = nullptr;
    cudaEvent_t bench_step_stop = nullptr;
    if (bench_mode && bench_eager) {
      check_cuda(cudaEventCreate(&bench_step_start), "cudaEventCreate(bench_step_start)");
      check_cuda(cudaEventCreate(&bench_step_stop), "cudaEventCreate(bench_step_stop)");
    }

    const bool run_generation = !bench_mode || bench_eager;
    const int num_candidates = static_cast<int>(first_token_choices.size());
    if (run_generation) {
    for (int candidate = 0; candidate < num_candidates; ++candidate) {
      if (candidate > 0) {
        reset_kv_allocator();
        run_prefill();
      }
      std::copy(block_table_snapshot.begin(), block_table_snapshot.end(), block_table);
      free_blocks_storage = free_blocks_snapshot;
      paged_attention_state.free_blocks = free_blocks_storage.data();
      paged_attention_state.free_blocks_count = free_blocks_count_snapshot;
      current_prompt_len[static_cast<size_t>(slot)] = prompt_len_snapshot;
      generated_tokens[static_cast<size_t>(slot)].clear();
      is_slot_free[static_cast<size_t>(slot)] = 0;

      std::mt19937 sample_rng(SAMPLE_SEED + static_cast<uint32_t>(candidate) * 10007u);
      double sum_logprob = 0.0;
      int decode_steps = 0;

      const int first_token = first_token_choices[static_cast<size_t>(candidate)].id;
      const double first_logprob = first_token_choices[static_cast<size_t>(candidate)].logprob;
      sum_logprob += first_logprob;
      ++decode_steps;
      generated_tokens[static_cast<size_t>(slot)].push_back(first_token);
      last_generated_tokens[static_cast<size_t>(slot)] = first_token;
      std::cout << "Best-of-N candidate " << candidate << " start (seed="
                << (SAMPLE_SEED + static_cast<uint32_t>(candidate) * 10007u)
                << " first_token=" << first_token << " first_logprob=" << first_logprob << ")\n";

      if (!bench_mode && is_stop_token(first_token)) {
        is_slot_free[static_cast<size_t>(slot)] = 1;
        generated_tokens[static_cast<size_t>(slot)].clear();  // stop tokens are not kept as answer text
        const double cand_score =
            score_generation_candidate(generated_tokens[static_cast<size_t>(slot)], sum_logprob, decode_steps);
        std::cout << "Best-of-N candidate " << candidate << ": tokens=0 decode_steps=" << decode_steps
                  << " avg_logprob=" << (sum_logprob / static_cast<double>(decode_steps))
                  << " score=" << cand_score << " (immediate EOT)\n";
        if (cand_score > best_score) {
          best_score = cand_score;
          best_tokens = generated_tokens[static_cast<size_t>(slot)];
          best_candidate = candidate;
        }
        continue;
      }

    while (!is_slot_free[static_cast<size_t>(slot)] &&
           (bench_mode ? bench_forwards < bench_steps
                       : generated_tokens[static_cast<size_t>(slot)].size() < static_cast<size_t>(run_max_new))) {
    const int decode_token_id = last_generated_tokens[static_cast<size_t>(slot)];
    const size_t decode_token_index = static_cast<size_t>(current_prompt_len[static_cast<size_t>(slot)]);
    // Row 0 of the activation buffers. The absolute position is only for RoPE and the KV page.
    const size_t decode_token_offset = 0;
    const size_t decode_mlp_offset = 0;
    if (decode_token_index >= static_cast<size_t>(MAX_SEQ_LEN)) {
      throw std::runtime_error("decode token index exceeds MAX_SEQ_LEN (" + std::to_string(MAX_SEQ_LEN) + ")");
    }

    if (bench_mode) {
      nvtxRangePushA("decode_eager");
      check_cuda(cudaEventRecord(bench_step_start), "cudaEventRecord(decode_eager)");
    }
    *decode_token_gpu = decode_token_id;
    llama_prefill::embedding_gather(decode_token_gpu, hidden_state + decode_token_offset, weights.tok_embeddings,
                                    static_cast<size_t>(num_decode_tokens));
    if (verbose_decode) {
      std::cout << "Gathered decode token " << decode_token_id << " at sequence index " << decode_token_index
                << " into " << static_cast<void*>(hidden_state + decode_token_offset) << '\n';
    }

    if (!bench_mode && !verified_first_decode_embed) {
      constexpr int DECODE_EMBED_VERIFY_DIMS = 8;
      std::vector<__nv_bfloat16> decode_embed_sample(DECODE_EMBED_VERIFY_DIMS);
      std::vector<__nv_bfloat16> expected_embed_sample(DECODE_EMBED_VERIFY_DIMS);
      check_cuda(cudaMemcpy(decode_embed_sample.data(), hidden_state + decode_token_offset,
                            decode_embed_sample.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(decode hidden_state sample D2H)");
      check_cuda(cudaMemcpy(expected_embed_sample.data(),
                            weights.tok_embeddings +
                                static_cast<size_t>(decode_token_id) * static_cast<size_t>(LLAMA_HIDDEN_SIZE),
                            expected_embed_sample.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(expected tok embedding sample D2H)");
      for (int dim = 0; dim < DECODE_EMBED_VERIFY_DIMS; ++dim) {
        const float actual = __bfloat162float(decode_embed_sample[static_cast<size_t>(dim)]);
        const float expected = __bfloat162float(expected_embed_sample[static_cast<size_t>(dim)]);
        if (std::fabs(actual - expected) > 1.0e-3f) {
          throw std::runtime_error("Decode token embedding mismatch at dim " + std::to_string(dim) + ": got " +
                                   std::to_string(actual) + ", expected " + std::to_string(expected));
        }
      }
      std::cout << "Decode token embedding verified for token " << decode_token_id << " at index "
                << decode_token_index << '\n';
      verified_first_decode_embed = true;
    }

    const int decode_seq_len = current_prompt_len[static_cast<size_t>(slot)] + 1;
    *gpu_seq_lens = decode_seq_len;

    for (size_t layer = 0; layer < num_layers; ++layer) {
      decode::rmsNorm(hidden_state + decode_token_offset, rms_norms + decode_token_offset,
                      weights.rms_attn[layer], num_decode_tokens);

      // q proj (1, 2048)
      check_cublas(cublasGemmEx(cublas_guard.handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                embedding_length,
                                num_decode_tokens,
                                embedding_length,
                                &q_proj_alpha,
                                weights.w_q[layer],
                                CUDA_R_16BF,
                                embedding_length,
                                rms_norms + decode_token_offset,
                                CUDA_R_16BF,
                                embedding_length,
                                &q_proj_beta,
                                q_proj + decode_token_offset,
                                CUDA_R_16BF,
                                embedding_length,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode q_proj)");

      // k proj (1, 512), writing output to k_proj_batched_buffer for scatter into K cache
      // K proj = rms_norms (1, 2048) * W_k (512, 2048)
      // W_k is stored as (512, 2048) (out features, in features), so we transpose it.
      // Row-major data appears transposed to cuBLAS (column-major), so GEMM is W_k^T * rms_norms^T
      // and the column-major output lands as row-major K_proj (1, 512).
      check_cublas(cublasGemmEx(cublas_guard.handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                kv_dim,
                                num_decode_tokens,
                                embedding_length,
                                &k_proj_alpha,
                                weights.w_k[layer],
                                CUDA_R_16BF,
                                embedding_length,
                                rms_norms + decode_token_offset,
                                CUDA_R_16BF,
                                embedding_length,
                                &k_proj_beta,
                                k_proj_batched_buffer,
                                CUDA_R_16BF,
                                kv_dim,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode k_proj)");

      // v proj (1, 512), writing output to v_proj_batched_buffer for scatter into V cache
      check_cublas(cublasGemmEx(cublas_guard.handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                kv_dim,
                                num_decode_tokens,
                                embedding_length,
                                &v_proj_alpha,
                                weights.w_v[layer],
                                CUDA_R_16BF,
                                embedding_length,
                                rms_norms + decode_token_offset,
                                CUDA_R_16BF,
                                embedding_length,
                                &v_proj_beta,
                                v_proj_batched_buffer,
                                CUDA_R_16BF,
                                kv_dim,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode v_proj)");
      decode::ropeDecode(q_proj + decode_token_offset, static_cast<int>(decode_token_index), embedding_length);
      decode::ropeDecode(k_proj_batched_buffer, static_cast<int>(decode_token_index), kv_dim);

      // Decode appends one token to the paged KV cache at sequence position current_prompt_len[slot].
      const int seq_len = current_prompt_len[slot];
      if (paged_attention_state.block_size <= 0) {
        throw std::runtime_error("paged attention requires positive block_size");
      }
      if (paged_attention_state.max_blocks_per_seq <= 0) {
        throw std::runtime_error("paged attention requires positive max_blocks_per_seq");
      }
      const int logical_block_idx = seq_len / paged_attention_state.block_size;
      const int token_in_block_idx = seq_len % paged_attention_state.block_size;
      if (logical_block_idx >= paged_attention_state.max_blocks_per_seq) {
        throw std::runtime_error("decode paged attention block_idx exceeds max_blocks_per_seq");
      }

      const size_t block_table_index =
          static_cast<size_t>(slot) * num_layers * paged_attention_state.max_blocks_per_seq +
          layer * paged_attention_state.max_blocks_per_seq + static_cast<size_t>(logical_block_idx);
      int block = block_table[block_table_index];
      if (token_in_block_idx == 0 && block == -1) {
        if (paged_attention_state.free_blocks_count == 0) {
          throw std::runtime_error("paged attention has no free blocks available for decode");
        }
        const size_t free_block_idx = paged_attention_state.free_blocks_count - 1;
        block = paged_attention_state.free_blocks[free_block_idx];
        paged_attention_state.free_blocks_count = free_block_idx;
        block_table[block_table_index] = block;
      }
      if (block == -1) {
        throw std::runtime_error("decode paged attention block was not allocated");
      }

      __nv_bfloat16* k_cache_ptr = reinterpret_cast<__nv_bfloat16*>(
          reinterpret_cast<char*>(paged_attention_state.kv_cache) + static_cast<size_t>(block) * paged_attention_state.block_bytes +
          static_cast<size_t>(token_in_block_idx) * static_cast<size_t>(kv_dim) * sizeof(__nv_bfloat16));
      const __nv_bfloat16* k_proj_ptr = k_proj_batched_buffer;
      check_cuda(cudaMemcpy(k_cache_ptr, k_proj_ptr, static_cast<size_t>(kv_dim) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
                 "cudaMemcpy(decode paged attention K)");

      __nv_bfloat16* v_cache_ptr = reinterpret_cast<__nv_bfloat16*>(
          reinterpret_cast<char*>(paged_attention_state.kv_cache) + static_cast<size_t>(block) * paged_attention_state.block_bytes +
          paged_attention_state.v_offset + static_cast<size_t>(token_in_block_idx) * static_cast<size_t>(kv_dim) * sizeof(__nv_bfloat16));
      const __nv_bfloat16* v_proj_ptr = v_proj_batched_buffer;
      check_cuda(cudaMemcpy(v_cache_ptr, v_proj_ptr, static_cast<size_t>(kv_dim) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
                 "cudaMemcpy(decode paged attention V)");

      // Q is already at row 0. pagedAttention reads it into registers, then overwrites that row.
      decode::pagedAttention(static_cast<int>(layer),
                             1,
                             q_proj,
                             reinterpret_cast<__nv_bfloat16*>(paged_attention_state.kv_cache),
                             block_table,
                             gpu_seq_lens,
                             gpu_active_slots,
                             q_proj);

      // O proj = (attention output) * W_o^T -> (1, 2048)
      __nv_bfloat16* o_proj = rms_norms + decode_token_offset;
      check_cublas(cublasGemmEx(cublas_guard.handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                embedding_length,
                                num_decode_tokens,
                                embedding_length,
                                &o_proj_alpha,
                                weights.w_o[layer],
                                CUDA_R_16BF,
                                embedding_length,
                                q_proj,
                                CUDA_R_16BF,
                                embedding_length,
                                &o_proj_beta,
                                o_proj,
                                CUDA_R_16BF,
                                embedding_length,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode o_proj)");
      decode::residualAdd(hidden_state + decode_token_offset, o_proj, num_active_slots);
      decode::rmsNorm(hidden_state + decode_token_offset, rms_norms + decode_token_offset,
                      weights.rms_ffn[layer], num_active_slots);

      // MLP gate proj = rms_norms (1, 2048) * W_gate^T -> (1, 8192)
      check_cublas(cublasGemmEx(cublas_guard.handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                mlp_intermediate_size,
                                num_active_slots,
                                embedding_length,
                                &gate_alpha,
                                weights.w_gate[layer],
                                CUDA_R_16BF,
                                embedding_length,
                                rms_norms + decode_token_offset,
                                CUDA_R_16BF,
                                embedding_length,
                                &gate_beta,
                                mlp_gate + decode_mlp_offset,
                                CUDA_R_16BF,
                                mlp_intermediate_size,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode mlp_gate)");

      // MLP up proj = rms_norms (1, 2048) * W_up^T -> (1, 8192)
      check_cublas(cublasGemmEx(cublas_guard.handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                mlp_intermediate_size,
                                num_active_slots,
                                embedding_length,
                                &up_alpha,
                                weights.w_up[layer],
                                CUDA_R_16BF,
                                embedding_length,
                                rms_norms + decode_token_offset,
                                CUDA_R_16BF,
                                embedding_length,
                                &up_beta,
                                mlp_up + decode_mlp_offset,
                                CUDA_R_16BF,
                                mlp_intermediate_size,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode mlp_up)");
      llama_prefill::silu(mlp_gate + decode_mlp_offset, mlp_up + decode_mlp_offset,
                          static_cast<size_t>(num_active_slots));

      // MLP down proj = silu(gate) * up (1, 8192) * W_down^T -> (1, 2048)
      __nv_bfloat16* down = q_proj + decode_token_offset;
      check_cublas(cublasGemmEx(cublas_guard.handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                embedding_length,
                                num_active_slots,
                                mlp_intermediate_size,
                                &down_alpha,
                                weights.w_down[layer],
                                CUDA_R_16BF,
                                mlp_intermediate_size,
                                mlp_gate + decode_mlp_offset,
                                CUDA_R_16BF,
                                mlp_intermediate_size,
                                &down_beta,
                                down,
                                CUDA_R_16BF,
                                embedding_length,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode mlp_down)");
      decode::residualAdd(hidden_state + decode_token_offset, down, num_active_slots);
    }

    decode::rmsNorm(hidden_state + decode_token_offset, rms_norms + decode_token_offset,
                    weights.final_norm, num_active_slots);

    // logits = rms_norm * E^T  (same layout as prefill LM head / tied embeddings)
    check_cublas(cublasGemmEx(cublas_guard.handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              vocab_size,
                              num_active_slots,
                              embedding_length,
                              &embed_alpha,
                              weights.tok_embeddings,
                              CUDA_R_16BF,
                              embedding_length,
                              rms_norms + decode_token_offset,
                              CUDA_R_16BF,
                              embedding_length,
                              &embed_beta,
                              embed_proj,
                              CUDA_R_16BF,
                              vocab_size,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode lm_head)");

    // Managed embed_proj: wait for lm_head, then sample on the host in place.
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(decode embed_proj)");
    if (bench_mode) {
      check_cuda(cudaEventRecord(bench_step_stop), "cudaEventRecord(decode_eager stop)");
      check_cuda(cudaEventSynchronize(bench_step_stop), "cudaEventSynchronize(decode_eager)");
      float step_ms = 0.0f;
      check_cuda(cudaEventElapsedTime(&step_ms, bench_step_start, bench_step_stop), "cudaEventElapsedTime(decode_eager)");
      bench_gpu_ms += static_cast<double>(step_ms);
      nvtxRangePop();
      nvtxRangePushA("sample_logits");
    }
    const auto sample_t0 = std::chrono::steady_clock::now();

    // Single active stream for now (num_active_slots == 1); layout matches multi-slot batching.
    const int active_slot = slot;
    std::vector<float> sampled_logits(static_cast<size_t>(vocab_size));
    for (int token_idx = 0; token_idx < vocab_size; ++token_idx) {
      sampled_logits[static_cast<size_t>(token_idx)] =
          __bfloat162float(embed_proj[static_cast<size_t>(token_idx)]);
    }
    double chosen_logprob = 0.0;
    const int max_token_idx =
        sample_token(sampled_logits, run_temperature, run_top_p, sample_rng, &chosen_logprob);
    const float max_token = sampled_logits[static_cast<size_t>(max_token_idx)];
    sum_logprob += chosen_logprob;
    ++decode_steps;
    if (bench_mode) {
      bench_sample_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sample_t0).count();
      bench_eager_outputs.push_back(max_token_idx);
      nvtxRangePop();
    }
    if (verbose_decode) {
      std::cout << "Output token: " << max_token << ", token index: " << max_token_idx
                << " logprob=" << chosen_logprob << '\n';
    }

    const bool hit_max_seq = current_prompt_len[static_cast<size_t>(active_slot)] == MAX_SEQ_LEN - 1;
    if (!bench_mode && (is_stop_token(max_token_idx) || hit_max_seq)) {
      is_slot_free[static_cast<size_t>(active_slot)] = 1;
      for (size_t layer = 0; layer < num_layers; ++layer) {
        for (int logical_block_idx = 0; logical_block_idx < paged_attention_state.max_blocks_per_seq;
             ++logical_block_idx) {
          const size_t block_idx =
              static_cast<size_t>(active_slot) * num_layers * paged_attention_state.max_blocks_per_seq +
              layer * paged_attention_state.max_blocks_per_seq + static_cast<size_t>(logical_block_idx);
          if (block_table[block_idx] != -1) {
            if (paged_attention_state.free_blocks_count >= free_blocks_storage.size()) {
              throw std::runtime_error("paged attention free_blocks overflow while releasing slot");
            }
            paged_attention_state.free_blocks[paged_attention_state.free_blocks_count++] = block_table[block_idx];
            block_table[block_idx] = -1;
          }
        }
      }
      if (verbose_decode) {
        std::cout << "Decode stopped (EOS/EOT/max len); freed slot " << active_slot << '\n';
      }
    } else {
      last_generated_tokens[static_cast<size_t>(active_slot)] = max_token_idx;
      generated_tokens[static_cast<size_t>(active_slot)].push_back(max_token_idx);
      current_prompt_len[static_cast<size_t>(active_slot)] =
          current_prompt_len[static_cast<size_t>(active_slot)] + 1;
      if (verbose_decode) {
        std::cout << "Decode continue: next_token=" << max_token_idx
                  << " prompt_len=" << current_prompt_len[static_cast<size_t>(active_slot)]
                  << " new_tokens=" << generated_tokens[static_cast<size_t>(active_slot)].size() << '\n';
      }
    }

    if (verbose_decode) {
      std::cout << "Decode forward pass completed: prompt_len=" << current_prompt_len[static_cast<size_t>(slot)]
                << " decode_token=" << decode_token_id << " layers=" << num_layers
                << " slot_free=" << static_cast<int>(is_slot_free[static_cast<size_t>(slot)]) << '\n';
    }
    if (bench_mode) {
      ++bench_forwards;
    }
    }  // while (!is_slot_free[slot] && generated < MAX_NEW_TOKENS)

    if (!is_slot_free[static_cast<size_t>(slot)]) {
      is_slot_free[static_cast<size_t>(slot)] = 1;
      for (size_t layer = 0; layer < num_layers; ++layer) {
        for (int logical_block_idx = 0; logical_block_idx < paged_attention_state.max_blocks_per_seq;
             ++logical_block_idx) {
          const size_t block_idx =
              static_cast<size_t>(slot) * num_layers * paged_attention_state.max_blocks_per_seq +
              layer * paged_attention_state.max_blocks_per_seq + static_cast<size_t>(logical_block_idx);
          if (block_table[block_idx] != -1) {
            if (paged_attention_state.free_blocks_count >= free_blocks_storage.size()) {
              throw std::runtime_error("paged attention free_blocks overflow while releasing slot");
            }
            paged_attention_state.free_blocks[paged_attention_state.free_blocks_count++] = block_table[block_idx];
            block_table[block_idx] = -1;
          }
        }
      }
      if (verbose_decode) {
        std::cout << "Decode stopped (max_new_tokens=" << run_max_new << "); freed slot " << slot << '\n';
      }
    }

      const std::vector<int>& cand_tokens = generated_tokens[static_cast<size_t>(slot)];
      const double cand_score = score_generation_candidate(cand_tokens, sum_logprob, decode_steps);
      const double avg_lp =
          decode_steps > 0 ? (sum_logprob / static_cast<double>(decode_steps)) : -1.0e9;
      std::cout << "Best-of-N candidate " << candidate << ": tokens=" << cand_tokens.size()
                << " decode_steps=" << decode_steps << " avg_logprob=" << avg_lp
                << " score=" << cand_score << '\n';
      if (cand_score > best_score) {
        best_score = cand_score;
        best_tokens = cand_tokens;
        best_candidate = candidate;
      }
    }  // for candidate
    }  // run_generation

    if (eval_mode) {
      const EvalItem& item = eval_items[eval_pass];
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_t0).count();
      nlohmann::json row;
      row["id"] = item.id;
      row["tokens"] = best_tokens;
      row["prompt_tokens"] = item.tokens.size();
      row["seconds"] = seconds;
      eval_out << row.dump() << '\n';
      eval_out.flush();
      std::cout << "eval " << (eval_pass + 1) << '/' << eval_passes << " id=" << item.id
                << " prompt=" << item.tokens.size() << " new=" << best_tokens.size() << " sec=" << seconds << '\n'
                << std::flush;
    }

    if (bench_mode) {
      std::cout << std::fixed;
      std::cout.precision(3);
      const double prefill_tokens_per_s =
          prefill_ms > 0.0f ? (static_cast<double>(gpu_input_tokens.count) * 1000.0 / static_cast<double>(prefill_ms))
                            : 0.0;
      std::cout << "\n== Benchmark ==\n";
      std::cout << "prompt_tokens=" << gpu_input_tokens.count << " layers=" << num_layers
                << " decode_steps=" << bench_steps << " sampling=greedy warmup=" << (bench_warmup ? 1 : 0) << '\n';
      std::cout << "prefill_ms=" << prefill_ms << " prefill_tokens_per_s=" << prefill_tokens_per_s << '\n';
      if (bench_eager) {
        if (bench_forwards != bench_steps) {
          throw std::runtime_error("eager benchmark ran " + std::to_string(bench_forwards) + " steps, expected " +
                                   std::to_string(bench_steps));
        }
        const double gpu_per = bench_gpu_ms / static_cast<double>(bench_steps);
        const double sample_per = bench_sample_ms / static_cast<double>(bench_steps);
        const double e2e_per = gpu_per + sample_per;
        std::cout << "eager: gpu_ms_per_token=" << gpu_per << " tokens_per_s_gpu=" << (1000.0 / gpu_per)
                  << " sample_ms_per_token=" << sample_per << " e2e_ms_per_token=" << e2e_per
                  << " tokens_per_s=" << (1000.0 / e2e_per) << '\n';
        std::cout << "eager tokens:";
        const size_t shown = std::min<size_t>(bench_eager_outputs.size(), 12);
        for (size_t i = 0; i < shown; ++i) {
          std::cout << ' ' << bench_eager_outputs[i];
        }
        if (bench_eager_outputs.size() > shown) {
          std::cout << " ...";
        }
        std::cout << '\n';
      }

      DecodeBenchBuffers bench_buffers;
      bench_buffers.weights = prefill_weights;
      bench_buffers.paged = &paged_attention_state;
      bench_buffers.free_blocks = free_blocks_storage.data();
      bench_buffers.block_table = block_table;
      bench_buffers.block_table_elems = block_table_elems;
      bench_buffers.num_layers = static_cast<int>(num_layers);
      bench_buffers.vocab_size = vocab_size;
      bench_buffers.prompt_len = prompt_len_snapshot;
      bench_buffers.hidden_state = hidden_state;
      bench_buffers.rms_norms = rms_norms;
      bench_buffers.q_proj = q_proj;
      bench_buffers.k_proj = k_proj_batched_buffer;
      bench_buffers.v_proj = v_proj_batched_buffer;
      bench_buffers.mlp_gate = mlp_gate;
      bench_buffers.mlp_up = mlp_up;
      bench_buffers.logits = embed_proj;
      bench_buffers.kv_snapshot = kv_snapshot;
      bench_buffers.kv_cache_bytes = kv_cache_bytes;
      bench_buffers.block_table_snapshot = &block_table_snapshot;
      bench_buffers.free_blocks_snapshot = &free_blocks_snapshot;
      bench_buffers.free_blocks_count_snapshot = free_blocks_count_snapshot;

      if (bench_async || bench_graph) {
        reset_kv_allocator();
        run_prefill();
      }
      const int bench_first_token = first_token_choices.front().id;
      const std::vector<int>* expected_tokens = bench_eager ? &bench_eager_outputs : nullptr;
      DecodeBenchResult async_result;
      if (bench_async) {
        async_result = bench_async_decode(bench_buffers, bench_steps, bench_first_token);
        if (!report_decode_bench("async", async_result, bench_steps, expected_tokens)) {
          bench_tokens_match = false;
        }
        if (!bench_eager) {
          expected_tokens = &async_result.outputs;
        }
      }
      if (bench_graph) {
        const DecodeBenchResult graph_result = bench_graph_decode(bench_buffers, bench_steps, bench_first_token);
        if (!report_decode_bench("cuda_graph", graph_result, bench_steps, expected_tokens)) {
          bench_tokens_match = false;
        }
        if (bench_eager && graph_result.gpu_ms > 0.0) {
          const double eager_per = bench_gpu_ms / static_cast<double>(bench_steps);
          const double graph_per = graph_result.gpu_ms / static_cast<double>(bench_steps);
          std::cout << "cuda_graph speedup vs eager (gpu)=" << (eager_per / graph_per) << '\n';
        }
        if (bench_async && async_result.gpu_ms > 0.0 && graph_result.gpu_ms > 0.0) {
          std::cout << "cuda_graph speedup vs async (gpu)=" << (async_result.gpu_ms / graph_result.gpu_ms) << '\n';
        }
      }
      if (bench_step_start != nullptr) {
        check_cuda(cudaEventDestroy(bench_step_start), "cudaEventDestroy(bench_step_start)");
        check_cuda(cudaEventDestroy(bench_step_stop), "cudaEventDestroy(bench_step_stop)");
      }
    }

    if (kv_snapshot != nullptr) {
      check_cuda(cudaFree(kv_snapshot), "cudaFree(kv_snapshot)");
    }
    if (!bench_mode && !eval_mode) {
    generated_tokens[static_cast<size_t>(slot)] = best_tokens;
    std::cout << "Best-of-N selected candidate " << best_candidate << " with score=" << best_score << '\n';

    std::cout << "Generated tokens:";
    for (int token_id : generated_tokens[static_cast<size_t>(slot)]) {
      std::cout << ' ' << token_id;
    }
    std::cout << '\n';
    }
    }  // eval_pass

    check_cuda(cudaFreeHost(gpu_seq_lens), "cudaFreeHost(gpu_seq_lens)");
    check_cuda(cudaFreeHost(gpu_active_slots), "cudaFreeHost(gpu_active_slots)");
    check_cuda(cudaFreeHost(decode_token_gpu), "cudaFreeHost(decode_token_gpu)");

    if (!bench_mode && !eval_mode) {
      print_input_embedding_debug(gpu_input_tokens, input_embeddings);
      print_mapping_debug(weights);
    }

    check_cuda(cudaFree(paged_attention_state.kv_cache), "cudaFree(paged_attention_state.kv_cache)");
    check_cuda(cudaFreeHost(block_table), "cudaFreeHost(block_table)");
    check_cuda(cudaFreeHost(embed_proj), "cudaFreeHost(embed_proj)");
    check_cuda(cudaFree(prefill_attn_scores), "cudaFree(prefill_attn_scores)");
    check_cuda(cudaFree(mlp_up), "cudaFree(mlp_up)");
    check_cuda(cudaFree(mlp_gate), "cudaFree(mlp_gate)");
    check_cuda(cudaFree(v_proj_batched_buffer), "cudaFree(v_proj_batched_buffer)");
    check_cuda(cudaFree(k_proj_batched_buffer), "cudaFree(k_proj_batched_buffer)");
    check_cuda(cudaFree(q_proj), "cudaFree(q_proj)");
    check_cuda(cudaFree(rms_norms), "cudaFree(rms_norms)");
    check_cuda(cudaFree(hidden_state), "cudaFree(hidden_state)");
    check_cuda(cudaFree(model_weights), "cudaFree(model_weights)");
    check_cuda(cudaFree(input_embeddings), "cudaFree(input_embeddings)");
    if (bench_mode && !bench_tokens_match) {
      std::cerr << "error: CUDA graph/async decode tokens did not match eager decode\n";
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
