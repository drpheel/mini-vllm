#pragma once

#include "prefill.cuh"

#include <cstddef>
#include <vector>

// Post-prefill buffers and the KV snapshot the benchmark restores between runs.
struct DecodeBenchBuffers {
  llama_prefill::PrefillWeights weights;
  llama_prefill::PagedAttentionState* paged = nullptr;
  int* free_blocks = nullptr;
  int* block_table = nullptr;
  size_t block_table_elems = 0;
  int num_layers = 0;
  int vocab_size = 0;
  int prompt_len = 0;
  __nv_bfloat16* hidden_state = nullptr;
  __nv_bfloat16* rms_norms = nullptr;
  __nv_bfloat16* q_proj = nullptr;
  __nv_bfloat16* k_proj = nullptr;
  __nv_bfloat16* v_proj = nullptr;
  __nv_bfloat16* mlp_gate = nullptr;
  __nv_bfloat16* mlp_up = nullptr;
  __nv_bfloat16* logits = nullptr;
  void* kv_snapshot = nullptr;
  size_t kv_cache_bytes = 0;
  const std::vector<int>* block_table_snapshot = nullptr;
  const std::vector<int>* free_blocks_snapshot = nullptr;
  size_t free_blocks_count_snapshot = 0;
};

struct DecodeBenchResult {
  std::vector<int> outputs;
  double gpu_ms = 0.0;
  double sample_ms = 0.0;
  double wall_ms = 0.0;
  int graph_nodes = 0;
};

// Same math as the eager decode loop, but every launch is asynchronous on one stream
// and the current token always occupies row 0. KV destinations and the RoPE position
// are device memory, so the launch sequence can be captured once and replayed.
DecodeBenchResult bench_async_decode(DecodeBenchBuffers& buffers, int steps, int first_token);
DecodeBenchResult bench_graph_decode(DecodeBenchBuffers& buffers, int steps, int first_token);

// Prints a timing line. Returns false when `expected` is set and the token ids differ.
bool report_decode_bench(const char* name,
                         const DecodeBenchResult& result,
                         int steps,
                         const std::vector<int>* expected);
