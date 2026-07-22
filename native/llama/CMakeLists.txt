cmake_minimum_required(VERSION 3.20)

project(clap-llama LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CLAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(CLAP_LLAMA_VENDOR_DIR "${CLAP_ROOT}/vendor/llama.cpp")
set(CLAP_CACHE_DIR "${CLAP_ROOT}/native/cache")

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
)

add_executable(clap-llama ${CLAP_LLAMA_SOURCES})
add_dependencies(clap-llama clap-cache-ffi-build)
target_include_directories(clap-llama PRIVATE "${CLAP_LLAMA_VENDOR_DIR}/vendor")
target_link_libraries(clap-llama PRIVATE llama clap-cache-ffi)

install(TARGETS clap-llama
  RUNTIME DESTINATION .
  COMPONENT runtime
)

if(BUILD_TESTING)
  add_executable(clap-llama-characterization native-characterization.test.cpp)
  add_test(NAME clap-llama-characterization COMMAND clap-llama-characterization)
endif()
