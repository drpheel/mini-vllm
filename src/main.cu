#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#include "decode.cuh"
#include "prompt_tokens.cuh"
#include "prefill.cuh"

#include <algorithm>
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
constexpr int LLAMA_HIDDEN_SIZE = llama_prefill::HIDDEN_SIZE;
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
    __nv_bfloat16* hidden_state = nullptr;
    __nv_bfloat16* rms_norms = nullptr;
    __nv_bfloat16* q_proj = nullptr;
    __nv_bfloat16* k_proj_batched_buffer = nullptr;
    __nv_bfloat16* v_proj_batched_buffer = nullptr;
    __nv_bfloat16* mlp_gate = nullptr;
    __nv_bfloat16* mlp_up = nullptr;
    __nv_bfloat16* prefill_attn_scores = nullptr;
    __nv_bfloat16* embed_proj = nullptr;
    const size_t kv_proj_bytes =
        static_cast<size_t>(prompt_tokens::MAX_PROMPT_LEN) * llama_prefill::KV_DIM * sizeof(__nv_bfloat16);
    const size_t mlp_intermediate_bytes =
        static_cast<size_t>(prompt_tokens::MAX_PROMPT_LEN) * llama_prefill::MLP_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16);
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

    const size_t num_layers = prefill_weights.input_layernorm.size();
    const size_t prompt_len = gpu_input_tokens.count;
    const size_t prefill_scores_elements =
        num_layers * static_cast<size_t>(llama_prefill::NUM_Q_HEADS) * prompt_len * prompt_len;
    const size_t prefill_scores_bytes = std::max(prefill_scores_elements * sizeof(__nv_bfloat16), sizeof(__nv_bfloat16));
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&prefill_attn_scores), prefill_scores_bytes),
               "cudaMalloc(prefill_attn_scores)");
    const size_t embed_proj_bytes =
        prompt_len * static_cast<size_t>(prefill_weights.vocab_size) * sizeof(__nv_bfloat16);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&embed_proj), embed_proj_bytes), "cudaMalloc(embed_proj)");
    std::vector<__nv_bfloat16> embed_proj_cpu(prompt_len * static_cast<size_t>(prefill_weights.vocab_size));

    llama_prefill::PagedAttentionState paged_attention_state;
    paged_attention_state.slot = 0;
    paged_attention_state.block_size = 16;
    paged_attention_state.max_blocks_per_seq = llama_prefill::MAX_BLOCKS_PER_SEQ;
    const int total_blocks = std::max(1, static_cast<int>(num_layers) * paged_attention_state.max_blocks_per_seq);
    const size_t per_kv_cache_bytes =
        static_cast<size_t>(paged_attention_state.block_size) * llama_prefill::KV_DIM * sizeof(__nv_bfloat16);
    paged_attention_state.v_offset = per_kv_cache_bytes;
    paged_attention_state.block_bytes = per_kv_cache_bytes * 2;
    check_cuda(cudaMalloc(&paged_attention_state.kv_cache,
                          static_cast<size_t>(total_blocks) * paged_attention_state.block_bytes),
               "cudaMalloc(paged_attention_state.kv_cache)");

    std::vector<int> block_table(
        static_cast<size_t>(llama_prefill::MAX_SEQUENCES) * llama_prefill::N_LAYERS * llama_prefill::MAX_BLOCKS_PER_SEQ,
        -1);
    int* block_table_gpu = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&block_table_gpu),
                          llama_prefill::MAX_SEQUENCES * llama_prefill::N_LAYERS * llama_prefill::MAX_BLOCKS_PER_SEQ *
                              sizeof(int)),
               "cudaMalloc(block_table_gpu)");
    std::vector<int> free_blocks_storage(static_cast<size_t>(total_blocks));
    std::iota(free_blocks_storage.begin(), free_blocks_storage.end(), 0);
    paged_attention_state.block_table = block_table.data();
    paged_attention_state.free_blocks = free_blocks_storage.data();
    paged_attention_state.free_blocks_count = free_blocks_storage.size();
    std::vector<std::vector<int>> generated_tokens(1);
    std::vector<int> last_generated_tokens(1);
    std::vector<int> current_prompt_len(1);
    llama_prefill::prefill(gpu_input_tokens.device_ptr, gpu_input_tokens.count, input_embeddings, hidden_state, rms_norms, q_proj,
                           k_proj_batched_buffer, v_proj_batched_buffer, mlp_gate, mlp_up, prefill_weights,
                           &paged_attention_state, prefill_attn_scores, embed_proj, embed_proj_cpu.data(),
                           generated_tokens, last_generated_tokens, current_prompt_len, block_table, block_table_gpu);
    std::cout << "Gathered " << gpu_input_tokens.count << " token embeddings into "
              << static_cast<void*>(input_embeddings) << '\n';
    const int slot = paged_attention_state.slot;
    const int decode_token_id = last_generated_tokens[slot];
    const size_t decode_token_index = static_cast<size_t>(current_prompt_len[slot]);
    const size_t decode_token_offset = decode_token_index * static_cast<size_t>(LLAMA_HIDDEN_SIZE);
    const size_t decode_mlp_offset = decode_token_index * static_cast<size_t>(llama_prefill::MLP_INTERMEDIATE_SIZE);
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
    const int kv_dim = llama_prefill::KV_DIM;

    int* decode_token_gpu = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&decode_token_gpu), sizeof(int)), "cudaMalloc(decode_token_gpu)");
    check_cuda(cudaMemcpy(decode_token_gpu, &decode_token_id, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(decode_token_gpu H2D)");
    llama_prefill::embedding_gather(decode_token_gpu, hidden_state + decode_token_offset, weights.tok_embeddings,
                                    static_cast<size_t>(num_decode_tokens));
    std::cout << "Gathered decode token " << decode_token_id << " at sequence index " << decode_token_index
              << " into " << static_cast<void*>(hidden_state + decode_token_offset) << '\n';

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

    CublasHandleGuard cublas_guard;
    check_cublas(cublasCreate(&cublas_guard.handle), "cublasCreate(decode)");

    int* gpu_seq_lens = nullptr;
    int* gpu_active_slots = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_seq_lens), sizeof(int)), "cudaMalloc(gpu_seq_lens)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&gpu_active_slots), sizeof(int)), "cudaMalloc(gpu_active_slots)");
    const int decode_seq_len = current_prompt_len[slot] + 1;
    check_cuda(cudaMemcpy(gpu_seq_lens, &decode_seq_len, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(gpu_seq_lens H2D)");
    check_cuda(cudaMemcpy(gpu_active_slots, &slot, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(gpu_active_slots H2D)");

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

      // Synchronize block table on CPU with block table on GPU (for attention).
      check_cuda(cudaMemcpy(block_table_gpu,
                            block_table.data(),
                            llama_prefill::MAX_SEQUENCES * llama_prefill::N_LAYERS *
                                llama_prefill::MAX_BLOCKS_PER_SEQ * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(decode block_table H2D)");

      // pagedAttention expects q/output at active_slot * HIDDEN_SIZE; copy decode Q to slot 0.
      check_cuda(cudaMemcpy(q_proj,
                            q_proj + decode_token_offset,
                            static_cast<size_t>(embedding_length) * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice),
                 "cudaMemcpy(decode q_proj to slot 0)");
      decode::pagedAttention(static_cast<int>(layer),
                             1,
                             q_proj,
                             reinterpret_cast<__nv_bfloat16*>(paged_attention_state.kv_cache),
                             block_table_gpu,
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

    std::cout << "Decode forward pass completed: prompt_len=" << current_prompt_len[slot]
              << " decode_token=" << decode_token_id << " layers=" << num_layers << '\n';

    check_cuda(cudaFree(gpu_seq_lens), "cudaFree(gpu_seq_lens)");
    check_cuda(cudaFree(gpu_active_slots), "cudaFree(gpu_active_slots)");
    check_cuda(cudaFree(decode_token_gpu), "cudaFree(decode_token_gpu)");

    print_input_embedding_debug(gpu_input_tokens, input_embeddings);
    print_mapping_debug(weights);

    check_cuda(cudaFree(paged_attention_state.kv_cache), "cudaFree(paged_attention_state.kv_cache)");
    check_cuda(cudaFree(block_table_gpu), "cudaFree(block_table_gpu)");
    check_cuda(cudaFree(embed_proj), "cudaFree(embed_proj)");
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
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
