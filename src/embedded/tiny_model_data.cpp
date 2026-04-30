// This file is the default stub for the embedded EUPE tiny model.
// Run scripts/embed_tiny_model.sh <path-to-eupe_vit_t16_fp16.onnx> to
// regenerate this file with the real model bytes. When no model is
// embedded, EUPE_TINY_MODEL_SIZE is 0 and the bundled-model APIs
// report that no tiny model is available.

#include "embedded_model.hpp"

namespace duckdb {

const uint8_t EUPE_TINY_MODEL_DATA[1] = {0};
const size_t EUPE_TINY_MODEL_SIZE = 0;

} // namespace duckdb
