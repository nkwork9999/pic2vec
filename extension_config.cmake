# This file is included by DuckDB's build system. It specifies which extension to load

# DuckDB 1.5.x defaults core targets to C++11. MSVC 19.51 needs C++17 for
# DuckDB's vendored fmt checked-iterator path.
if(MSVC)
    set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to enforce" FORCE)
    set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require selected C++ standard" FORCE)
endif()

# Extension from this repo
duckdb_extension_load(pic2vec
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)
