if(NOT DEFINED TEST_EXECUTABLE)
  message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()

if(NOT DEFINED MODEL_PATH)
  message(FATAL_ERROR "MODEL_PATH is required")
endif()

if(NOT DEFINED PROMPT_PATH)
  message(FATAL_ERROR "PROMPT_PATH is required")
endif()

if(NOT EXISTS "${MODEL_PATH}")
  message("Skipping llama model + prompt GPU smoke test: model not found at ${MODEL_PATH}")
  return()
endif()

file(WRITE "${PROMPT_PATH}" "791 6864 315 9822 374\n")

execute_process(
  COMMAND "${TEST_EXECUTABLE}" "${MODEL_PATH}" "${PROMPT_PATH}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message("stdout:\n${stdout}")
  message("stderr:\n${stderr}")
  message(FATAL_ERROR "llama model + prompt GPU smoke test failed with exit code ${result}")
endif()

if(NOT stdout MATCHES "Allocated prompt input embeddings")
  message("stdout:\n${stdout}")
  message("stderr:\n${stderr}")
  message(FATAL_ERROR "llama model + prompt GPU smoke test did not allocate input embeddings")
endif()

if(NOT stdout MATCHES "Gathered 5 token embeddings into")
  message("stdout:\n${stdout}")
  message("stderr:\n${stderr}")
  message(FATAL_ERROR "llama model + prompt GPU smoke test did not run embedding gather kernel")
endif()

if(NOT stdout MATCHES "Decode token embedding verified")
  message("stdout:\n${stdout}")
  message("stderr:\n${stderr}")
  message(FATAL_ERROR "llama model + prompt GPU smoke test did not verify decode token embedding")
endif()

message("${stdout}")
