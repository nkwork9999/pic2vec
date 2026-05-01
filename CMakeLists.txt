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
# ---------------------------------------------------------------------------
set(EXTENSION_SOURCES
    src/pic2vec_extension.cpp
    src/embedded/tiny_model_data.cpp
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

set_target_properties(${EXTENSION_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(${EXTENSION_NAME} PRIVATE -std=c++17)
target_compile_options(${LOADABLE_EXTENSION_NAME} PRIVATE -std=c++17)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
