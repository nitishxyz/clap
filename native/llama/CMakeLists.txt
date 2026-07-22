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
  clap-llama.cpp
  src/environment.cpp
  src/protocol.cpp
  src/stop-buffer.cpp
)

set(CLAP_LLAMA_HEADERS
  include/clap/llama/environment.h
  include/clap/llama/protocol.h
  include/clap/llama/stop-buffer.h
)

add_executable(clap-llama ${CLAP_LLAMA_SOURCES} ${CLAP_LLAMA_HEADERS})
add_dependencies(clap-llama clap-cache-ffi-build)
target_include_directories(clap-llama PRIVATE
  "${CLAP_LLAMA_INCLUDE_DIR}"
  "${CLAP_LLAMA_VENDOR_DIR}/vendor"
)
target_link_libraries(clap-llama PRIVATE llama clap-cache-ffi)

install(TARGETS clap-llama
  RUNTIME DESTINATION .
  COMPONENT runtime
)

if(BUILD_TESTING)
  add_executable(clap-llama-characterization native-characterization.test.cpp)
  add_test(NAME clap-llama-characterization COMMAND clap-llama-characterization)

  add_executable(clap-llama-environment
    tests/environment.test.cpp
    src/environment.cpp
  )
  target_include_directories(clap-llama-environment PRIVATE "${CLAP_LLAMA_INCLUDE_DIR}")
  add_test(NAME clap-llama-environment COMMAND clap-llama-environment)

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
  )
  add_test(NAME clap-llama-protocol COMMAND clap-llama-protocol)

  add_executable(clap-llama-stop-buffer
    tests/stop-buffer.test.cpp
    src/stop-buffer.cpp
  )
  target_include_directories(clap-llama-stop-buffer PRIVATE "${CLAP_LLAMA_INCLUDE_DIR}")
  add_test(NAME clap-llama-stop-buffer COMMAND clap-llama-stop-buffer)

  add_custom_target(clap-llama-tests DEPENDS
    clap-llama-characterization
    clap-llama-environment
    clap-llama-protocol
    clap-llama-stop-buffer
  )
endif()
