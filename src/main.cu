#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#include "prompt_tokens.cuh"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
constexpr int LLAMA_HIDDEN_SIZE = 2048;
constexpr const char* DEFAULT_MODEL_PATH = "/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors";

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
  }
}

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

__global__ void embeddingGatherKernel(int* gpu_input_tokens,
                                      __nv_bfloat16* input_embeddings,
                                      __nv_bfloat16* embed_tokens) {
  int workIndex = threadIdx.x + blockIdx.x * 2048;
  input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
  input_embeddings[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
}

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

void gather_input_embeddings(const prompt_tokens::GpuPromptTokens& gpu_input_tokens,
                             __nv_bfloat16* input_embeddings,
                             const ModelWeights& weights) {
  if (gpu_input_tokens.count == 0) {
    return;
  }

  embeddingGatherKernel<<<static_cast<unsigned int>(gpu_input_tokens.count), 1024>>>(
      gpu_input_tokens.device_ptr, input_embeddings, weights.tok_embeddings);
  check_cuda(cudaGetLastError(), "embeddingGatherKernel launch");
  check_cuda(cudaDeviceSynchronize(), "embeddingGatherKernel");

  std::cout << "Gathered " << gpu_input_tokens.count << " token embeddings into "
            << static_cast<void*>(input_embeddings) << '\n';
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
    if (argc > 3) {
      throw std::runtime_error("usage: cuda_cublas_json [model.safetensors] [prompt_tokens.txt]");
    }

    const std::string model_path = argc > 1 ? argv[1] : DEFAULT_MODEL_PATH;
    const std::string prompt_tokens_path = argc > 2 ? argv[2] : "";

    print_gpu_status();
    __nv_bfloat16* input_embeddings = nullptr;
    const size_t input_embeddings_bytes =
        static_cast<size_t>(prompt_tokens::MAX_PROMPT_LEN) * LLAMA_HIDDEN_SIZE * sizeof(__nv_bfloat16);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&input_embeddings), input_embeddings_bytes),
               "cudaMalloc(input_embeddings)");
    std::cout << "Allocated prompt input embeddings at " << static_cast<void*>(input_embeddings) << " ("
              << format_bytes(input_embeddings_bytes) << ")\n";

    prompt_tokens::GpuPromptTokens gpu_input_tokens;
    if (!prompt_tokens_path.empty()) {
      gpu_input_tokens = prompt_tokens::load_token_ids_to_gpu(prompt_tokens_path);
      std::cout << "Loaded " << gpu_input_tokens.count << " prompt tokens to GPU at "
                << static_cast<void*>(gpu_input_tokens.device_ptr) << '\n';
    }

    std::cout << "\nLoading safetensors file: " << model_path << '\n';

    const SafetensorsFile metadata = read_safetensors_metadata(model_path);
    std::cout << "Header bytes: " << metadata.header_len << " (" << format_bytes(metadata.header_len) << ")\n";
    std::cout << "Payload start: " << metadata.payload_start << '\n';
    std::cout << "Payload bytes: " << metadata.payload_size << " (" << format_bytes(metadata.payload_size) << ")\n";
    std::cout << "Tensor count: " << metadata.tensors.size() << '\n';

    void* model_weights = nullptr;
    check_cuda(cudaMalloc(&model_weights, metadata.payload_size), "cudaMalloc(model_weights)");
    std::cout << "Allocated one GPU block at " << model_weights << '\n';

    copy_payload_to_gpu(metadata, model_weights);
    ModelWeights weights = build_llama_weight_views(metadata, model_weights);
    gather_input_embeddings(gpu_input_tokens, input_embeddings, weights);
    print_input_embedding_debug(gpu_input_tokens, input_embeddings);
    print_mapping_debug(weights);

    check_cuda(cudaFree(model_weights), "cudaFree(model_weights)");
    check_cuda(cudaFree(input_embeddings), "cudaFree(input_embeddings)");
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
