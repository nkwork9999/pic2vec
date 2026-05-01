cmake_minimum_required(VERSION 3.5...3.29)

set(TARGET_NAME pic2vec)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Rust core library (tract-onnx + image preprocessing) via Corrosion
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(
    MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml"
    CRATES "pic2vec_core"
    PROFILE release
)

# ---------------------------------------------------------------------------
# Extension sources (no ONNX Runtime, no stb_image - all moved to Rust)
#
# tiny_model_data.cpp holds the bundled EUPE ViT-T ONNX bytes (~22MB raw,
# ~131MB as C++ array). It is .gitignored — too large to commit. CI/CD and
# fresh checkouts get an auto-generated stub here (size=0, model not bundled);
# users who want the zero-setup bundled model run scripts/embed_tiny_model.sh
# to overwrite tiny_model_data.cpp with the real bytes.
# ---------------------------------------------------------------------------
set(_PIC2VEC_TINY_DATA "${CMAKE_CURRENT_SOURCE_DIR}/src/embedded/tiny_model_data.cpp")
if(NOT EXISTS "${_PIC2VEC_TINY_DATA}")
    set(_PIC2VEC_TINY_DATA "${CMAKE_CURRENT_BINARY_DIR}/tiny_model_data_stub.cpp")
    if(NOT EXISTS "${_PIC2VEC_TINY_DATA}")
        file(WRITE "${_PIC2VEC_TINY_DATA}"
"// Auto-generated stub. Run scripts/embed_tiny_model.sh <onnx> to embed a real model.\n#include \"embedded_model.hpp\"\nnamespace duckdb {\nconst uint8_t EUPE_TINY_MODEL_DATA[1] = {0};\nconst size_t EUPE_TINY_MODEL_SIZE = 0;\n} // namespace duckdb\n")
    endif()
    message(STATUS "pic2vec: bundled tiny model not present; generated stub at ${_PIC2VEC_TINY_DATA}")
else()
    message(STATUS "pic2vec: using bundled tiny model from ${_PIC2VEC_TINY_DATA}")
endif()

set(EXTENSION_SOURCES
    src/pic2vec_extension.cpp
    "${_PIC2VEC_TINY_DATA}"
)

include_directories(src/include)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link Rust static library into the extension.
# DuckDB's build_*_extension uses plain target_link_libraries internally, so
# we must too. $<BUILD_INTERFACE:...> stops Corrosion's IMPORTED target from
# leaking into install(EXPORT "DuckDBExports") downstream.
# Corrosion exposes a staticlib crate as `<name>-static` (the real archive
# target) plus a top-level INTERFACE wrapper `<name>`. Linking the -static
# target by file path keeps it out of install(EXPORT "DuckDBExports").
if(TARGET pic2vec_core-static)
    set(_PIC2VEC_RUST_TARGET pic2vec_core-static)
else()
    set(_PIC2VEC_RUST_TARGET pic2vec_core)
endif()

add_dependencies(${EXTENSION_NAME} ${_PIC2VEC_RUST_TARGET})
add_dependencies(${LOADABLE_EXTENSION_NAME} ${_PIC2VEC_RUST_TARGET})
target_link_libraries(${EXTENSION_NAME} $<TARGET_FILE:${_PIC2VEC_RUST_TARGET}>)
target_link_libraries(${LOADABLE_EXTENSION_NAME} $<TARGET_FILE:${_PIC2VEC_RUST_TARGET}>)

if(APPLE)
    target_link_libraries(${EXTENSION_NAME} "-framework CoreFoundation" "-framework Security")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework CoreFoundation" "-framework Security")
endif()

# Rust std on Windows pulls in OS-level libs that the MSVC link line bundles
# automatically but MinGW/rtools doesn't:
#   - ntdll: NtReadFile/NtWriteFile (std::fs)
#   - userenv: std::env::home_dir
#   - bcrypt: BCryptGenRandom (getrandom crate, used transitively by tract)
if(WIN32)
    target_link_libraries(${EXTENSION_NAME} ntdll userenv bcrypt)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} ntdll userenv bcrypt)
endif()

# Force C++17 (DuckDB defaults to C++11; std::bitset, std::sqrt overloads,
# iomanip etc. need C++17). CXX_STANDARD maps to the right per-compiler flag
# automatically (no explicit -std=c++17 needed; that style trips MSVC).
set_target_properties(${EXTENSION_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
