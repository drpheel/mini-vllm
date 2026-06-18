cmake_minimum_required(VERSION 3.22)

project(cuda_cublas_json_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(CUDAToolkit REQUIRED)
include(CTest)

set(LLAMA_MODEL_PATH "/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors"
  CACHE FILEPATH "Safetensors file used by the llama model + prompt GPU smoke test")

add_executable(cuda_cublas_json
  src/main.cu
  src/prefill.cu
)

target_link_libraries(cuda_cublas_json
  PRIVATE
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)

if(BUILD_TESTING)
  add_executable(prompt_tokens_test
    tests/prompt_tokens_test.cu
  )

  target_link_libraries(prompt_tokens_test
    PRIVATE
      CUDA::cudart
  )

  add_test(NAME prompt_tokens_test COMMAND prompt_tokens_test)
  set_tests_properties(prompt_tokens_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(prefill_test
    tests/prefill_test.cu
    src/prefill.cu
  )

  target_link_libraries(prefill_test
    PRIVATE
      CUDA::cudart
  )

  add_test(NAME prefill_test COMMAND prefill_test)
  set_tests_properties(prefill_test PROPERTIES SKIP_RETURN_CODE 77)

  add_test(
    NAME llama_model_prompt_gpu_smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DTEST_EXECUTABLE=$<TARGET_FILE:cuda_cublas_json>
        -DMODEL_PATH=${LLAMA_MODEL_PATH}
        -DPROMPT_PATH=${CMAKE_CURRENT_BINARY_DIR}/llama_prompt_tokens.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/llama_prompt_smoke.cmake
  )
  set_tests_properties(llama_model_prompt_gpu_smoke PROPERTIES
    SKIP_REGULAR_EXPRESSION "Skipping llama model \\+ prompt GPU smoke test")
endif()
