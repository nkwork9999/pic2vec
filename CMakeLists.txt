cmake_minimum_required(VERSION 3.5...3.29)

set(TARGET_NAME pic2vec)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ONNX Runtime - locate Homebrew/system install (no CMake config provided by brew).
# Override with -DONNXRUNTIME_ROOT=/path if installed elsewhere.
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(EXISTS "/opt/homebrew/include/onnxruntime/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_ROOT "/opt/homebrew")
    elseif(EXISTS "/usr/local/include/onnxruntime/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_ROOT "/usr/local")
    endif()
endif()

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNX Runtime not found. Install via: brew install onnxruntime, or set -DONNXRUNTIME_ROOT=/path")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIB}")

set(EXTENSION_SOURCES
    src/pic2vec_extension.cpp
    src/onnx_inference.cpp
    src/image_utils.cpp
    src/embedded/tiny_model_data.cpp
)

include_directories(src/include ${ONNXRUNTIME_INCLUDE_DIR})

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

target_link_libraries(${EXTENSION_NAME} ${ONNXRUNTIME_LIB})
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${ONNXRUNTIME_LIB})

# ONNX Runtime headers and std::make_unique require C++17.
# Override DuckDB's default C++11 setting for our targets only.
set_target_properties(${EXTENSION_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(${EXTENSION_NAME} PRIVATE -std=c++17)
target_compile_options(${LOADABLE_EXTENSION_NAME} PRIVATE -std=c++17)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
