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
# Bundled tiny model embedding
#
# Single source of truth: assets/eupe_vit_t16_fp16.onnx (committed, ~22MB).
# scripts/embed_tiny_model.sh converts it to a C++ byte array under the build
# tree (${CMAKE_CURRENT_BINARY_DIR}/generated/tiny_model_data.cpp, ~131MB).
# Nothing is written into the source tree, so no .gitignore juggling is needed.
#
# If the .onnx is missing or the embed script fails for any reason, the build
# is aborted with FATAL_ERROR. There is intentionally no stub fallback: a
# silently-empty model produced a binary that loaded but crashed at first
# pic_embed() call, masking CI breakage on community-extensions.duckdb.org.
# ---------------------------------------------------------------------------
set(_PIC2VEC_TINY_ONNX   "${CMAKE_CURRENT_SOURCE_DIR}/assets/eupe_vit_t16_fp16.onnx")
set(_PIC2VEC_EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_tiny_model.sh")
set(_PIC2VEC_TINY_DATA   "${CMAKE_CURRENT_BINARY_DIR}/generated/tiny_model_data.cpp")

if(NOT EXISTS "${_PIC2VEC_TINY_ONNX}")
    message(FATAL_ERROR
        "pic2vec: bundled model not found at ${_PIC2VEC_TINY_ONNX}. "
        "The .onnx is the only source of truth for the bundled EUPE ViT-T model "
        "and must be committed to the repository.")
endif()
if(NOT EXISTS "${_PIC2VEC_EMBED_SCRIPT}")
    message(FATAL_ERROR
        "pic2vec: embed script missing at ${_PIC2VEC_EMBED_SCRIPT}.")
endif()

# Generate at configure time so that downstream `add_library` / source listing
# sees a real file (CMake doesn't auto-create OUTPUTs of add_custom_command at
# configure time). Re-runs whenever the .onnx or the script changes via the
# add_custom_command rule below.
message(STATUS "pic2vec: embedding bundled model from ${_PIC2VEC_TINY_ONNX}")
execute_process(
    COMMAND bash "${_PIC2VEC_EMBED_SCRIPT}" "${_PIC2VEC_TINY_ONNX}" "${_PIC2VEC_TINY_DATA}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE _PIC2VEC_EMBED_RC
    OUTPUT_VARIABLE _PIC2VEC_EMBED_OUT
    ERROR_VARIABLE  _PIC2VEC_EMBED_ERR
)
if(NOT _PIC2VEC_EMBED_RC EQUAL 0)
    message(FATAL_ERROR
        "pic2vec: embed_tiny_model.sh failed (rc=${_PIC2VEC_EMBED_RC}).\n"
        "stdout: ${_PIC2VEC_EMBED_OUT}\n"
        "stderr: ${_PIC2VEC_EMBED_ERR}")
endif()
if(NOT EXISTS "${_PIC2VEC_TINY_DATA}")
    message(FATAL_ERROR
        "pic2vec: embed_tiny_model.sh returned 0 but did not produce ${_PIC2VEC_TINY_DATA}.\n"
        "stdout: ${_PIC2VEC_EMBED_OUT}\n"
        "stderr: ${_PIC2VEC_EMBED_ERR}")
endif()
message(STATUS "pic2vec: bundled model embedded at ${_PIC2VEC_TINY_DATA}")

# Re-run the embed at build time when the .onnx or the script changes, so an
# updated model triggers a recompile of just tiny_model_data.cpp.
add_custom_command(
    OUTPUT  "${_PIC2VEC_TINY_DATA}"
    COMMAND bash "${_PIC2VEC_EMBED_SCRIPT}" "${_PIC2VEC_TINY_ONNX}" "${_PIC2VEC_TINY_DATA}"
    DEPENDS "${_PIC2VEC_TINY_ONNX}" "${_PIC2VEC_EMBED_SCRIPT}"
    COMMENT "pic2vec: re-embedding bundled EUPE ViT-T model"
    VERBATIM
)

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
