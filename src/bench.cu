#include "bench.cuh"

#include "decode.cuh"

#include <cublas_v2.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

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

void* alloc_mapped(size_t bytes, const char* what) {
  void* host_ptr = nullptr;
  check_cuda(cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped), what);
  void* device_ptr = nullptr;
  check_cuda(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0), "cudaHostGetDevicePointer");
  if (device_ptr != host_ptr) {
    cudaFreeHost(host_ptr);
    throw std::runtime_error(std::string(what) + ": expected a unified host/device pointer");
  }
  return host_ptr;
}

struct HostFree {
  void* ptr = nullptr;
  ~HostFree() {
    if (ptr != nullptr) {
      cudaFreeHost(ptr);
    }
  }
};

struct DeviceFree {
  void* ptr = nullptr;
  ~DeviceFree() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

struct StreamFree {
  cudaStream_t stream = nullptr;
  ~StreamFree() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

struct EventFree {
  cudaEvent_t event = nullptr;
  ~EventFree() {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }
};

struct CublasFree {
  cublasHandle_t handle = nullptr;
  ~CublasFree() {
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
  }
};

struct GraphFree {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ~GraphFree() {
    if (exec != nullptr) {
      cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
  }
};

void restore_snapshot(DecodeBenchBuffers& buffers) {
  // A null snapshot means the caller already rebuilt the prompt cache with prefill.
  if (buffers.kv_snapshot != nullptr) {
    check_cuda(cudaMemcpy(buffers.paged->kv_cache, buffers.kv_snapshot, buffers.kv_cache_bytes, cudaMemcpyDeviceToDevice),
               "cudaMemcpy(bench kv restore)");
  }
  if (buffers.block_table_snapshot->size() != buffers.block_table_elems ||
      buffers.free_blocks_snapshot->size() < buffers.free_blocks_count_snapshot) {
    throw std::runtime_error("KV snapshot size does not match the live cache");
  }
  std::copy(buffers.block_table_snapshot->begin(), buffers.block_table_snapshot->end(), buffers.block_table);
  std::copy(buffers.free_blocks_snapshot->begin(), buffers.free_blocks_snapshot->end(), buffers.free_blocks);
  buffers.paged->free_blocks = buffers.free_blocks;
  buffers.paged->free_blocks_count = buffers.free_blocks_count_snapshot;
}

void check_layout(const DecodeBenchBuffers& buffers) {
  // pagedAttentionKernel indexes the block table with compile-time N_LAYERS.
  // With slot 0 that term drops out, so a checkpoint with fewer layers still matches
  // as long as each layer's pages sit at layer * MAX_BLOCKS_PER_SEQ.
  if (buffers.num_layers <= 0 || buffers.num_layers > decode::N_LAYERS) {
    throw std::runtime_error("decode layer count is outside the compiled block table");
  }
  if (buffers.paged->slot != 0 && buffers.num_layers != decode::N_LAYERS) {
    throw std::runtime_error("non-zero KV slot requires the checkpoint layer count to match N_LAYERS");
  }
  if (buffers.paged->block_size != decode::BLOCK_SIZE ||
      buffers.paged->max_blocks_per_seq != decode::MAX_BLOCKS_PER_SEQ ||
      buffers.paged->block_bytes != static_cast<size_t>(decode::BLOCK_BYTES) ||
      buffers.paged->v_offset != static_cast<size_t>(decode::V_OFFSET)) {
    throw std::runtime_error("paged KV layout does not match decode kernel constants");
  }
  if (buffers.prompt_len <= 0 || buffers.vocab_size <= 0) {
    throw std::runtime_error("benchmark requires a non-empty prompt and vocab");
  }
}

// Host-side page allocation. The GPU launch that follows only reads the pointers written here.
void bind_decode_slot(DecodeBenchBuffers& buffers,
                      int position,
                      int token_id,
                      int* position_ptr,
                      int* seq_len_ptr,
                      int* token_ptr,
                      __nv_bfloat16** k_dst,
                      __nv_bfloat16** v_dst) {
  auto* paged = buffers.paged;
  if (paged->block_size <= 0) {
    throw std::runtime_error("paged attention requires positive block_size");
  }
  const int logical = position / paged->block_size;
  const int token_in_block = position % paged->block_size;
  if (logical >= paged->max_blocks_per_seq) {
    throw std::runtime_error("decode position exceeds the paged KV capacity");
  }

  *position_ptr = position;
  *seq_len_ptr = position + 1;
  *token_ptr = token_id;

  for (int layer = 0; layer < buffers.num_layers; ++layer) {
    const size_t index = static_cast<size_t>(paged->slot) * static_cast<size_t>(buffers.num_layers) *
                             static_cast<size_t>(paged->max_blocks_per_seq) +
                         static_cast<size_t>(layer) * static_cast<size_t>(paged->max_blocks_per_seq) +
                         static_cast<size_t>(logical);
    int block = buffers.block_table[index];
    if (token_in_block == 0 && block == -1) {
      if (paged->free_blocks_count == 0) {
        throw std::runtime_error("paged attention has no free blocks available for decode");
      }
      const size_t free_idx = paged->free_blocks_count - 1;
      block = paged->free_blocks[free_idx];
      paged->free_blocks_count = free_idx;
      buffers.block_table[index] = block;
    }
    if (block < 0) {
      throw std::runtime_error("decode paged attention block was not allocated");
    }

    auto* base = reinterpret_cast<char*>(paged->kv_cache) + static_cast<size_t>(block) * paged->block_bytes;
    const size_t row_bytes = static_cast<size_t>(token_in_block) * llama_prefill::KV_DIM * sizeof(__nv_bfloat16);
    k_dst[layer] = reinterpret_cast<__nv_bfloat16*>(base + row_bytes);
    v_dst[layer] = reinterpret_cast<__nv_bfloat16*>(base + paged->v_offset + row_bytes);
  }
}

int greedy_token(const __nv_bfloat16* logits, int vocab) {
  int best = 0;
  float best_value = __bfloat162float(logits[0]);
  for (int i = 1; i < vocab; ++i) {
    const float value = __bfloat162float(logits[i]);
    if (value > best_value) {
      best_value = value;
      best = i;
    }
  }
  return best;
}

// One decode token. Activations stay at row 0; history lives only in the KV cache.
// Nothing in this function synchronizes, so it can run inside cudaStreamBeginCapture.
void launch_decode_step(cublasHandle_t handle,
                        cudaStream_t stream,
                        DecodeBenchBuffers& buffers,
                        const int* token_ptr,
                        const int* position_ptr,
                        int* seq_len_ptr,
                        int* active_slot_ptr,
                        __nv_bfloat16* const* k_dst,
                        __nv_bfloat16* const* v_dst) {
  static const float alpha = 1.0f;
  static const float beta = 0.0f;
  constexpr int kHidden = llama_prefill::HIDDEN_SIZE;
  constexpr int kKv = llama_prefill::KV_DIM;
  constexpr int kMlp = llama_prefill::MLP_INTERMEDIATE_SIZE;

  auto gemm = [&](int m, int n, int k, const __nv_bfloat16* a, int lda, const __nv_bfloat16* b, int ldb,
                  __nv_bfloat16* c, int ldc) {
    check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb,
                              &beta, c, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(decode step)");
  };

  __nv_bfloat16* hidden = buffers.hidden_state;
  __nv_bfloat16* rms = buffers.rms_norms;
  __nv_bfloat16* q = buffers.q_proj;
  __nv_bfloat16* k = buffers.k_proj;
  __nv_bfloat16* v = buffers.v_proj;
  __nv_bfloat16* gate = buffers.mlp_gate;
  __nv_bfloat16* up = buffers.mlp_up;

  llama_prefill::embedding_gather_async(token_ptr, hidden, buffers.weights.tok_embeddings, 1, stream);

  for (int layer = 0; layer < buffers.num_layers; ++layer) {
    decode::rmsNorm(hidden, rms, buffers.weights.input_layernorm[static_cast<size_t>(layer)], 1, stream);
    gemm(kHidden, 1, kHidden, buffers.weights.w_q[static_cast<size_t>(layer)], kHidden, rms, kHidden, q, kHidden);
    gemm(kKv, 1, kHidden, buffers.weights.w_k[static_cast<size_t>(layer)], kHidden, rms, kHidden, k, kKv);
    gemm(kKv, 1, kHidden, buffers.weights.w_v[static_cast<size_t>(layer)], kHidden, rms, kHidden, v, kKv);
    decode::ropeDecodeFromDevice(q, position_ptr, kHidden, decode::HEAD_DIM, stream);
    decode::ropeDecodeFromDevice(k, position_ptr, kKv, decode::HEAD_DIM, stream);
    decode::scatterKvIndirect(k_dst, k, layer, kKv, stream);
    decode::scatterKvIndirect(v_dst, v, layer, kKv, stream);
    decode::pagedAttention(layer, 1, q, static_cast<__nv_bfloat16*>(buffers.paged->kv_cache), buffers.block_table,
                           seq_len_ptr, active_slot_ptr, q, stream);
    gemm(kHidden, 1, kHidden, buffers.weights.w_o[static_cast<size_t>(layer)], kHidden, q, kHidden, rms, kHidden);
    decode::residualAdd(hidden, rms, 1, stream);
    decode::rmsNorm(hidden, rms, buffers.weights.post_attention_layernorm[static_cast<size_t>(layer)], 1, stream);
    gemm(kMlp, 1, kHidden, buffers.weights.w_gate[static_cast<size_t>(layer)], kHidden, rms, kHidden, gate, kMlp);
    gemm(kMlp, 1, kHidden, buffers.weights.w_up[static_cast<size_t>(layer)], kHidden, rms, kHidden, up, kMlp);
    llama_prefill::silu_async(gate, up, 1, stream);
    gemm(kHidden, 1, kMlp, buffers.weights.w_down[static_cast<size_t>(layer)], kMlp, gate, kMlp, q, kHidden);
    decode::residualAdd(hidden, q, 1, stream);
  }

  decode::rmsNorm(hidden, rms, buffers.weights.norm, 1, stream);
  gemm(buffers.vocab_size, 1, kHidden, buffers.weights.tok_embeddings, kHidden, rms, kHidden, buffers.logits,
       buffers.vocab_size);
}

struct StepTimers {
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int* position = nullptr;
  int* seq_len = nullptr;
  int* token = nullptr;
  int* active_slot = nullptr;
  __nv_bfloat16** k_dst = nullptr;
  __nv_bfloat16** v_dst = nullptr;
  cublasHandle_t handle = nullptr;
};

DecodeBenchResult run_steps(DecodeBenchBuffers& buffers,
                            StepTimers& timers,
                            int steps,
                            int first_token,
                            const char* nvtx_name,
                            const std::function<void()>& launch) {
  DecodeBenchResult result;
  result.outputs.reserve(static_cast<size_t>(steps));
  int token = first_token;
  int position = buffers.prompt_len;
  for (int step = 0; step < steps; ++step) {
    const auto wall_start = std::chrono::steady_clock::now();
    bind_decode_slot(buffers, position, token, timers.position, timers.seq_len, timers.token, timers.k_dst, timers.v_dst);
    nvtxRangePushA(nvtx_name);
    check_cuda(cudaEventRecord(timers.start, timers.stream), "cudaEventRecord(bench start)");
    launch();
    check_cuda(cudaEventRecord(timers.stop, timers.stream), "cudaEventRecord(bench stop)");
    check_cuda(cudaEventSynchronize(timers.stop), "cudaEventSynchronize(bench stop)");
    nvtxRangePop();
    float gpu_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_ms, timers.start, timers.stop), "cudaEventElapsedTime(bench)");
    result.gpu_ms += static_cast<double>(gpu_ms);

    const auto sample_start = std::chrono::steady_clock::now();
    nvtxRangePushA("sample_logits");
    const int next = greedy_token(buffers.logits, buffers.vocab_size);
    nvtxRangePop();
    result.sample_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sample_start).count();
    result.wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
    result.outputs.push_back(next);
    token = next;
    ++position;
  }
  return result;
}

void prepare_cublas(cublasHandle_t* handle, void** workspace, cudaStream_t stream) {
  check_cublas(cublasCreate(handle), "cublasCreate(bench)");
  check_cublas(cublasSetStream(*handle, stream), "cublasSetStream(bench)");
  constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;
  check_cuda(cudaMalloc(workspace, kWorkspaceBytes), "cudaMalloc(cublas workspace)");
  check_cublas(cublasSetWorkspace(*handle, *workspace, kWorkspaceBytes), "cublasSetWorkspace(bench)");
}

}  // namespace

DecodeBenchResult bench_async_decode(DecodeBenchBuffers& buffers, int steps, int first_token) {
  check_layout(buffers);
  StreamFree stream;
  check_cuda(cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking), "cudaStreamCreate(bench async)");
  CublasFree cublas;
  DeviceFree workspace;
  prepare_cublas(&cublas.handle, &workspace.ptr, stream.stream);

  HostFree position;
  HostFree seq_len;
  HostFree token;
  HostFree active_slot;
  HostFree k_dst;
  HostFree v_dst;
  position.ptr = alloc_mapped(sizeof(int), "bench position");
  seq_len.ptr = alloc_mapped(sizeof(int), "bench seq_len");
  token.ptr = alloc_mapped(sizeof(int), "bench token");
  active_slot.ptr = alloc_mapped(sizeof(int), "bench active_slot");
  k_dst.ptr = alloc_mapped(static_cast<size_t>(buffers.num_layers) * sizeof(__nv_bfloat16*), "bench k_dst");
  v_dst.ptr = alloc_mapped(static_cast<size_t>(buffers.num_layers) * sizeof(__nv_bfloat16*), "bench v_dst");
  *static_cast<int*>(active_slot.ptr) = buffers.paged->slot;

  EventFree start;
  EventFree stop;
  check_cuda(cudaEventCreate(&start.event), "cudaEventCreate(async start)");
  check_cuda(cudaEventCreate(&stop.event), "cudaEventCreate(async stop)");

  // One real step so cuBLAS finishes lazy init before any timed replay.
  restore_snapshot(buffers);
  bind_decode_slot(buffers, buffers.prompt_len, first_token, static_cast<int*>(position.ptr), static_cast<int*>(seq_len.ptr),
                   static_cast<int*>(token.ptr), static_cast<__nv_bfloat16**>(k_dst.ptr),
                   static_cast<__nv_bfloat16**>(v_dst.ptr));
  nvtxRangePushA("decode_async_warmup");
  launch_decode_step(cublas.handle, stream.stream, buffers, static_cast<int*>(token.ptr), static_cast<int*>(position.ptr),
                     static_cast<int*>(seq_len.ptr), static_cast<int*>(active_slot.ptr),
                     static_cast<__nv_bfloat16**>(k_dst.ptr), static_cast<__nv_bfloat16**>(v_dst.ptr));
  check_cuda(cudaStreamSynchronize(stream.stream), "cudaStreamSynchronize(async warmup)");
  nvtxRangePop();
  restore_snapshot(buffers);

  StepTimers timers;
  timers.stream = stream.stream;
  timers.start = start.event;
  timers.stop = stop.event;
  timers.position = static_cast<int*>(position.ptr);
  timers.seq_len = static_cast<int*>(seq_len.ptr);
  timers.token = static_cast<int*>(token.ptr);
  timers.active_slot = static_cast<int*>(active_slot.ptr);
  timers.k_dst = static_cast<__nv_bfloat16**>(k_dst.ptr);
  timers.v_dst = static_cast<__nv_bfloat16**>(v_dst.ptr);
  timers.handle = cublas.handle;

  return run_steps(buffers, timers, steps, first_token, "decode_async", [&]() {
    launch_decode_step(timers.handle, timers.stream, buffers, timers.token, timers.position, timers.seq_len,
                       timers.active_slot, timers.k_dst, timers.v_dst);
  });
}

DecodeBenchResult bench_graph_decode(DecodeBenchBuffers& buffers, int steps, int first_token) {
  check_layout(buffers);
  StreamFree stream;
  check_cuda(cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking), "cudaStreamCreate(bench graph)");
  CublasFree cublas;
  DeviceFree workspace;
  prepare_cublas(&cublas.handle, &workspace.ptr, stream.stream);

  HostFree position;
  HostFree seq_len;
  HostFree token;
  HostFree active_slot;
  HostFree k_dst;
  HostFree v_dst;
  position.ptr = alloc_mapped(sizeof(int), "graph position");
  seq_len.ptr = alloc_mapped(sizeof(int), "graph seq_len");
  token.ptr = alloc_mapped(sizeof(int), "graph token");
  active_slot.ptr = alloc_mapped(sizeof(int), "graph active_slot");
  k_dst.ptr = alloc_mapped(static_cast<size_t>(buffers.num_layers) * sizeof(__nv_bfloat16*), "graph k_dst");
  v_dst.ptr = alloc_mapped(static_cast<size_t>(buffers.num_layers) * sizeof(__nv_bfloat16*), "graph v_dst");
  *static_cast<int*>(active_slot.ptr) = buffers.paged->slot;

  EventFree start;
  EventFree stop;
  check_cuda(cudaEventCreate(&start.event), "cudaEventCreate(graph start)");
  check_cuda(cudaEventCreate(&stop.event), "cudaEventCreate(graph stop)");

  auto launch = [&]() {
    launch_decode_step(cublas.handle, stream.stream, buffers, static_cast<int*>(token.ptr), static_cast<int*>(position.ptr),
                       static_cast<int*>(seq_len.ptr), static_cast<int*>(active_slot.ptr),
                       static_cast<__nv_bfloat16**>(k_dst.ptr), static_cast<__nv_bfloat16**>(v_dst.ptr));
  };

  restore_snapshot(buffers);
  bind_decode_slot(buffers, buffers.prompt_len, first_token, static_cast<int*>(position.ptr), static_cast<int*>(seq_len.ptr),
                   static_cast<int*>(token.ptr), static_cast<__nv_bfloat16**>(k_dst.ptr),
                   static_cast<__nv_bfloat16**>(v_dst.ptr));
  nvtxRangePushA("decode_graph_warmup");
  launch();
  check_cuda(cudaStreamSynchronize(stream.stream), "cudaStreamSynchronize(graph warmup)");
  nvtxRangePop();
  restore_snapshot(buffers);

  bind_decode_slot(buffers, buffers.prompt_len, first_token, static_cast<int*>(position.ptr), static_cast<int*>(seq_len.ptr),
                   static_cast<int*>(token.ptr), static_cast<__nv_bfloat16**>(k_dst.ptr),
                   static_cast<__nv_bfloat16**>(v_dst.ptr));
  check_cuda(cudaStreamBeginCapture(stream.stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture(decode)");
  launch();
  GraphFree graph;
  check_cuda(cudaStreamEndCapture(stream.stream, &graph.graph), "cudaStreamEndCapture(decode)");
  check_cuda(cudaGraphInstantiateWithFlags(&graph.exec, graph.graph, 0), "cudaGraphInstantiate(decode)");
  size_t node_count = 0;
  check_cuda(cudaGraphGetNodes(graph.graph, nullptr, &node_count), "cudaGraphGetNodes(decode)");
  restore_snapshot(buffers);

  StepTimers timers;
  timers.stream = stream.stream;
  timers.start = start.event;
  timers.stop = stop.event;
  timers.position = static_cast<int*>(position.ptr);
  timers.seq_len = static_cast<int*>(seq_len.ptr);
  timers.token = static_cast<int*>(token.ptr);
  timers.active_slot = static_cast<int*>(active_slot.ptr);
  timers.k_dst = static_cast<__nv_bfloat16**>(k_dst.ptr);
  timers.v_dst = static_cast<__nv_bfloat16**>(v_dst.ptr);
  timers.handle = cublas.handle;

  DecodeBenchResult result = run_steps(buffers, timers, steps, first_token, "decode_graph", [&]() {
    check_cuda(cudaGraphLaunch(graph.exec, timers.stream), "cudaGraphLaunch(decode)");
  });
  result.graph_nodes = static_cast<int>(node_count);
  return result;
}

bool report_decode_bench(const char* name,
                         const DecodeBenchResult& result,
                         int steps,
                         const std::vector<int>* expected) {
  if (steps <= 0 || result.outputs.size() != static_cast<size_t>(steps)) {
    throw std::runtime_error(std::string(name) + " benchmark did not record one token per step");
  }
  const double gpu_per = result.gpu_ms / static_cast<double>(steps);
  const double sample_per = result.sample_ms / static_cast<double>(steps);
  const double wall_per = result.wall_ms / static_cast<double>(steps);
  std::cout << std::fixed;
  std::cout.precision(3);
  std::cout << name << ": gpu_ms_per_token=" << gpu_per << " tokens_per_s_gpu=" << (1000.0 / gpu_per)
            << " sample_ms_per_token=" << sample_per << " e2e_ms_per_token=" << wall_per
            << " tokens_per_s=" << (1000.0 / wall_per);
  if (result.graph_nodes > 0) {
    std::cout << " graph_nodes=" << result.graph_nodes;
  }
  std::cout << '\n';
  std::cout << name << " tokens:";
  const size_t shown = std::min<size_t>(result.outputs.size(), 12);
  for (size_t i = 0; i < shown; ++i) {
    std::cout << ' ' << result.outputs[i];
  }
  if (result.outputs.size() > shown) {
    std::cout << " ...";
  }
  std::cout << '\n';

  if (expected == nullptr) {
    return true;
  }
  if (expected->size() != result.outputs.size()) {
    std::cout << name << " token count " << result.outputs.size() << " != expected " << expected->size() << '\n';
    return false;
  }
  for (size_t i = 0; i < result.outputs.size(); ++i) {
    if ((*expected)[i] != result.outputs[i]) {
      std::cout << name << " token mismatch at step " << i << ": got " << result.outputs[i] << ", expected "
                << (*expected)[i] << '\n';
      return false;
    }
  }
  std::cout << name << " tokens match the eager decode\n";
  return true;
}
