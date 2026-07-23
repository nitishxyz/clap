cmake_minimum_required(VERSION 3.20)

project(clap-llama LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CLAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(CLAP_LLAMA_VENDOR_DIR "${CLAP_ROOT}/vendor/llama.cpp")
set(CLAP_CACHE_DIR "${CLAP_ROOT}/native/cache")
set(CLAP_LLAMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(NOT EXISTS "${CLAP_LLAMA_VENDOR_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "llama.cpp is not vendored; run: bun run runtime:llama:vendor")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static llama.cpp libraries" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utilities" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "Enable llama.cpp CURL support" FORCE)

add_subdirectory("${CLAP_LLAMA_VENDOR_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

find_program(CLAP_CARGO_EXECUTABLE cargo REQUIRED)
if(WIN32)
  set(CLAP_CACHE_LIBRARY "${CLAP_CACHE_DIR}/target/release/clap_cache_ffi.lib")
else()
  set(CLAP_CACHE_LIBRARY "${CLAP_CACHE_DIR}/target/release/libclap_cache_ffi.a")
endif()

add_custom_target(clap-cache-ffi-build
  COMMAND "${CLAP_CARGO_EXECUTABLE}" build --release --locked -p clap-cache-ffi
  WORKING_DIRECTORY "${CLAP_CACHE_DIR}"
  BYPRODUCTS "${CLAP_CACHE_LIBRARY}"
  COMMENT "Building the Clap Rust cache library"
  VERBATIM
)

add_library(clap-cache-ffi STATIC IMPORTED GLOBAL)
set_target_properties(clap-cache-ffi PROPERTIES
  IMPORTED_LOCATION "${CLAP_CACHE_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CLAP_CACHE_DIR}/include"
)

# This is the sole explicit production source manifest. Add extracted worker
# translation units here as the native worker is decomposed.
set(CLAP_LLAMA_SOURCES
  src/cache-identity.cpp
  src/cache-executor.cpp
  src/environment.cpp
  src/generation-stepper.cpp
  src/main.cpp
  src/model-runtime.cpp
  src/prompt.cpp
  src/protocol.cpp
  src/request-state.cpp
  src/request-preparer.cpp
  src/sampling.cpp
  src/scheduler.cpp
  src/stop-buffer.cpp
  src/structured-output.cpp
  src/telemetry.cpp
  src/worker-state.cpp
  src/worker.cpp
)

set(CLAP_LLAMA_HEADERS
  include/clap/llama/cache-identity.h
  include/clap/llama/cache-executor.h
  include/clap/llama/environment.h
  include/clap/llama/generation-stepper.h
  include/clap/llama/model-runtime.h
  include/clap/llama/prompt.h
  include/clap/llama/protocol.h
  include/clap/llama/request-state.h
  include/clap/llama/request-preparer.h
  include/clap/llama/sampling.h
  include/clap/llama/scheduler.h
  include/clap/llama/stop-buffer.h
  include/clap/llama/structured-output.h
  include/clap/llama/telemetry.h
  include/clap/llama/worker-state.h
  include/clap/llama/worker.h
)

add_executable(clap-llama ${CLAP_LLAMA_SOURCES} ${CLAP_LLAMA_HEADERS})
add_dependencies(clap-llama clap-cache-ffi-build)
target_include_directories(clap-llama PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CLAP_LLAMA_INCLUDE_DIR}"
  "${CLAP_LLAMA_VENDOR_DIR}/vendor"
)
target_link_libraries(clap-llama PRIVATE llama llama-common clap-cache-ffi)

install(TARGETS clap-llama
  RUNTIME DESTINATION .
  COMPONENT runtime
)

if(BUILD_TESTING)
  add_executable(clap-llama-structured-output
    tests/structured-output.test.cpp
    src/protocol.cpp
    src/structured-output.cpp
  )
  target_include_directories(clap-llama-structured-output PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/common"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-structured-output PRIVATE llama llama-common)
  add_test(NAME clap-llama-structured-output COMMAND clap-llama-structured-output)

  add_executable(clap-llama-cache-identity
    tests/cache-identity.test.cpp
    src/cache-identity.cpp
  )
  add_dependencies(clap-llama-cache-identity clap-cache-ffi-build)
  target_include_directories(clap-llama-cache-identity PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-cache-identity PRIVATE llama clap-cache-ffi)
  target_compile_definitions(clap-llama-cache-identity PRIVATE
    CLAP_CACHE_IDENTITY_VECTOR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../packages/worker-protocol/fixtures/v1/cache-identity-vector.json"
  )
  add_test(NAME clap-llama-cache-identity COMMAND clap-llama-cache-identity)

  add_executable(clap-llama-worker
    tests/worker.test.cpp
    src/cache-identity.cpp
    src/cache-executor.cpp
    src/environment.cpp
    src/generation-stepper.cpp
    src/model-runtime.cpp
    src/prompt.cpp
    src/protocol.cpp
    src/request-preparer.cpp
    src/request-state.cpp
    src/sampling.cpp
    src/scheduler.cpp
    src/stop-buffer.cpp
    src/structured-output.cpp
    src/telemetry.cpp
    src/worker-state.cpp
    src/worker.cpp
  )
  add_dependencies(clap-llama-worker clap-cache-ffi-build)
  target_include_directories(clap-llama-worker PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-worker PRIVATE llama llama-common clap-cache-ffi)
  add_test(NAME clap-llama-worker COMMAND clap-llama-worker)

  add_executable(clap-llama-scheduler
    tests/scheduler.test.cpp
    src/cache-executor.cpp
    src/protocol.cpp
    src/request-state.cpp
    src/scheduler.cpp
    src/stop-buffer.cpp
  )
  add_dependencies(clap-llama-scheduler clap-cache-ffi-build)
  target_include_directories(clap-llama-scheduler PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-scheduler PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-scheduler COMMAND clap-llama-scheduler)

  add_executable(clap-llama-worker-state
    tests/worker-state.test.cpp
    src/cache-identity.cpp
    src/cache-executor.cpp
    src/environment.cpp
    src/generation-stepper.cpp
    src/model-runtime.cpp
    src/prompt.cpp
    src/protocol.cpp
    src/request-preparer.cpp
    src/request-state.cpp
    src/sampling.cpp
    src/stop-buffer.cpp
    src/structured-output.cpp
    src/telemetry.cpp
    src/worker-state.cpp
  )
  add_dependencies(clap-llama-worker-state clap-cache-ffi-build)
  target_include_directories(clap-llama-worker-state PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-worker-state PRIVATE llama llama-common clap-cache-ffi)
  add_test(NAME clap-llama-worker-state COMMAND clap-llama-worker-state)

  add_executable(clap-llama-generation-stepper
    tests/generation-stepper.test.cpp
    src/cache-executor.cpp
    src/environment.cpp
    src/generation-stepper.cpp
    src/model-runtime.cpp
    src/request-state.cpp
    src/stop-buffer.cpp
  )
  add_dependencies(clap-llama-generation-stepper clap-cache-ffi-build)
  target_include_directories(clap-llama-generation-stepper PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-generation-stepper PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-generation-stepper COMMAND clap-llama-generation-stepper)

  add_executable(clap-llama-cache-adapter cache-adapter.test.cpp)
  add_dependencies(clap-llama-cache-adapter clap-cache-ffi-build)
  target_link_libraries(clap-llama-cache-adapter PRIVATE clap-cache-ffi)
  add_test(NAME clap-llama-cache-adapter COMMAND clap-llama-cache-adapter)

  add_executable(clap-llama-cache-executor
    tests/cache-executor.test.cpp
    src/cache-executor.cpp
  )
  add_dependencies(clap-llama-cache-executor clap-cache-ffi-build)
  target_include_directories(clap-llama-cache-executor PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-cache-executor PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-cache-executor COMMAND clap-llama-cache-executor)

  add_executable(clap-llama-cache-checkpoint
    tests/cache-checkpoint.test.cpp
    src/cache-executor.cpp
  )
  add_dependencies(clap-llama-cache-checkpoint clap-cache-ffi-build)
  target_include_directories(clap-llama-cache-checkpoint PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-cache-checkpoint PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-cache-checkpoint COMMAND clap-llama-cache-checkpoint)
  set_tests_properties(clap-llama-cache-checkpoint PROPERTIES LABELS "tier-a;cache")

  add_executable(clap-llama-cache-transaction
    tests/cache-transaction.test.cpp
    src/cache-executor.cpp
  )
  add_dependencies(clap-llama-cache-transaction clap-cache-ffi-build)
  target_include_directories(clap-llama-cache-transaction PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-cache-transaction PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-cache-transaction COMMAND clap-llama-cache-transaction)
  set_tests_properties(clap-llama-cache-transaction PROPERTIES LABELS "tier-a;cache")

  add_executable(clap-llama-cache-request-lifecycle
    tests/cache-request-lifecycle.test.cpp
    src/cache-executor.cpp
    src/environment.cpp
    src/generation-stepper.cpp
    src/model-runtime.cpp
    src/request-state.cpp
    src/stop-buffer.cpp
  )
  add_dependencies(clap-llama-cache-request-lifecycle clap-cache-ffi-build)
  target_include_directories(clap-llama-cache-request-lifecycle PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-cache-request-lifecycle PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-cache-request-lifecycle COMMAND clap-llama-cache-request-lifecycle)
  set_tests_properties(clap-llama-cache-request-lifecycle PROPERTIES LABELS "tier-a;cache")

  add_executable(clap-llama-cache-model-probe
    tests/cache-model-probe.test.cpp
    src/cache-executor.cpp
    src/environment.cpp
    src/generation-stepper.cpp
    src/model-runtime.cpp
    src/request-state.cpp
    src/stop-buffer.cpp
  )
  add_dependencies(clap-llama-cache-model-probe clap-cache-ffi-build)
  target_include_directories(clap-llama-cache-model-probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-cache-model-probe PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-cache-model-probe COMMAND clap-llama-cache-model-probe)
  set_tests_properties(clap-llama-cache-model-probe PROPERTIES LABELS "tier-a;cache;probe")

  add_executable(clap-llama-characterization native-characterization.test.cpp)
  add_test(NAME clap-llama-characterization COMMAND clap-llama-characterization)

  add_executable(clap-llama-environment
    tests/environment.test.cpp
    src/environment.cpp
  )
  target_include_directories(clap-llama-environment PRIVATE "${CLAP_LLAMA_INCLUDE_DIR}")
  add_test(NAME clap-llama-environment COMMAND clap-llama-environment)

  add_executable(clap-llama-model-runtime
    tests/model-runtime.test.cpp
    src/environment.cpp
    src/model-runtime.cpp
  )
  target_include_directories(clap-llama-model-runtime PRIVATE
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-model-runtime PRIVATE llama)
  add_test(NAME clap-llama-model-runtime COMMAND clap-llama-model-runtime)

  add_executable(clap-llama-prompt
    tests/prompt.test.cpp
    src/environment.cpp
    src/model-runtime.cpp
    src/prompt.cpp
  )
  target_include_directories(clap-llama-prompt PRIVATE
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-prompt PRIVATE llama)
  add_test(NAME clap-llama-prompt COMMAND clap-llama-prompt)

  add_executable(clap-llama-protocol
    tests/protocol.test.cpp
    src/protocol.cpp
  )
  target_include_directories(clap-llama-protocol PRIVATE
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_compile_definitions(clap-llama-protocol PRIVATE
    CLAP_LLAMA_PROTOCOL_FIXTURE="${CLAP_ROOT}/packages/runtime-router/fixtures/legacy-worker-protocol/llama-events.jsonl"
    CLAP_LLAMA_PROTOCOL_V1_REQUEST_FIXTURE="${CLAP_ROOT}/packages/worker-protocol/fixtures/v1/requests/all.jsonl"
    CLAP_LLAMA_PROTOCOL_V1_EVENT_FIXTURE="${CLAP_ROOT}/packages/worker-protocol/fixtures/v1/events/all.jsonl"
  )
  add_test(NAME clap-llama-protocol COMMAND clap-llama-protocol)

  add_executable(clap-llama-request-state
    tests/request-state.test.cpp
    src/cache-executor.cpp
    src/request-state.cpp
    src/stop-buffer.cpp
  )
  add_dependencies(clap-llama-request-state clap-cache-ffi-build)
  target_include_directories(clap-llama-request-state PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-request-state PRIVATE llama clap-cache-ffi)
  add_test(NAME clap-llama-request-state COMMAND clap-llama-request-state)

  add_executable(clap-llama-request-preparer
    tests/request-preparer.test.cpp
    src/cache-identity.cpp
    src/cache-executor.cpp
    src/environment.cpp
    src/model-runtime.cpp
    src/prompt.cpp
    src/protocol.cpp
    src/request-preparer.cpp
    src/request-state.cpp
    src/sampling.cpp
    src/stop-buffer.cpp
    src/structured-output.cpp
  )
  add_dependencies(clap-llama-request-preparer clap-cache-ffi-build)
  target_include_directories(clap-llama-request-preparer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-request-preparer PRIVATE llama llama-common clap-cache-ffi)
  add_test(NAME clap-llama-request-preparer COMMAND clap-llama-request-preparer)

  add_executable(clap-llama-stop-buffer
    tests/stop-buffer.test.cpp
    src/stop-buffer.cpp
  )
  target_include_directories(clap-llama-stop-buffer PRIVATE "${CLAP_LLAMA_INCLUDE_DIR}")
  add_test(NAME clap-llama-stop-buffer COMMAND clap-llama-stop-buffer)

  add_executable(clap-llama-sampling
    tests/sampling.test.cpp
    src/sampling.cpp
  )
  target_include_directories(clap-llama-sampling PRIVATE
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-sampling PRIVATE llama)
  add_test(NAME clap-llama-sampling COMMAND clap-llama-sampling)

  add_executable(clap-llama-structured-sampling
    tests/structured-sampling.test.cpp
    src/cache-executor.cpp
    src/environment.cpp
    src/model-runtime.cpp
    src/protocol.cpp
    src/request-state.cpp
    src/sampling.cpp
    src/stop-buffer.cpp
    src/structured-output.cpp
  )
  add_dependencies(clap-llama-structured-sampling clap-cache-ffi-build)
  target_include_directories(clap-llama-structured-sampling PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/common"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_link_libraries(clap-llama-structured-sampling PRIVATE llama llama-common clap-cache-ffi)
  add_test(NAME clap-llama-structured-sampling COMMAND clap-llama-structured-sampling)

  add_executable(clap-llama-telemetry
    tests/telemetry.test.cpp
    src/telemetry.cpp
  )
  target_include_directories(clap-llama-telemetry PRIVATE
    "${CLAP_LLAMA_INCLUDE_DIR}"
    "${CLAP_LLAMA_VENDOR_DIR}/vendor"
  )
  target_compile_definitions(clap-llama-telemetry PRIVATE
    CLAP_LLAMA_TELEMETRY_FIXTURE="${CLAP_ROOT}/packages/runtime-router/fixtures/legacy-worker-protocol/llama-events.jsonl"
  )
  add_test(NAME clap-llama-telemetry COMMAND clap-llama-telemetry)

  add_custom_target(clap-llama-tests DEPENDS
    clap-llama-cache-adapter
    clap-llama-cache-checkpoint
    clap-llama-cache-identity
    clap-llama-cache-model-probe
    clap-llama-cache-executor
    clap-llama-cache-request-lifecycle
    clap-llama-cache-transaction
    clap-llama-characterization
    clap-llama-environment
    clap-llama-generation-stepper
    clap-llama-model-runtime
    clap-llama-prompt
    clap-llama-protocol
    clap-llama-request-state
    clap-llama-request-preparer
    clap-llama-sampling
    clap-llama-scheduler
    clap-llama-stop-buffer
    clap-llama-structured-sampling
    clap-llama-telemetry
    clap-llama-worker-state
    clap-llama-worker
  )
endif()
