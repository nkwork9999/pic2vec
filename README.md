# EUPE - DuckDB Extension

EUPE (Efficient Universal Perception Encoder) vision embeddings in DuckDB via ONNX Runtime.

Extract image embeddings, compute similarity, and build visual search — all from SQL.

## Quick Start

Three ways to get a model, in order of convenience:

### 1. Zero-setup (bundled tiny model)

If the extension was built with the ViT-T/16 model bundled in, just start calling
`eupe_embed()` — the bundled model auto-loads on first use:

```sql
LOAD eupe;
SELECT eupe_embed('/path/to/photo.jpg');  -- auto-loads bundled tiny model
```

Check whether a model was bundled:

```sql
SELECT eupe_bundled_info();
```

### 2. Download a larger model (whisper-style)

```sql
LOAD eupe;
SELECT eupe_download_model('vit_s16');   -- cached in ~/.duckdb/extensions/eupe/models/
SELECT eupe_embed('/path/to/photo.jpg', 'vit_s16');
```

Known models: `vit_t16`, `vit_s16`, `vit_b16`, `convnext_t`, `convnext_s`, `convnext_b`.

### 3. Load a local ONNX file

```sql
SELECT eupe_load_model('/path/to/my_eupe.onnx', 'my_model');
SELECT eupe_embed('/path/to/photo.jpg', 'my_model');
```

### Compute similarity

```sql
SELECT eupe_similarity(
    eupe_embed('image_a.jpg'),
    eupe_embed('image_b.jpg')
) AS similarity;
```

## Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `eupe_version()` | Extension version | VARCHAR |
| `eupe_bundled_info()` | Info about bundled tiny model (if any) | VARCHAR |
| `eupe_load_bundled()` | Explicitly load the bundled tiny model | VARCHAR |
| `eupe_download_model(name)` | Download known model to cache and load | VARCHAR |
| `eupe_load_model(path)` | Load ONNX model (name from filename) | VARCHAR |
| `eupe_load_model(path, name)` | Load ONNX model with name | VARCHAR |
| `eupe_unload_model(name)` | Unload model | VARCHAR |
| `eupe_list_models()` | List loaded models | VARCHAR |
| `eupe_embed(image_path)` | Extract embedding (default model, auto-loads bundled tiny) | LIST(FLOAT) |
| `eupe_embed(image_path, model)` | Extract embedding (named model) | LIST(FLOAT) |
| `eupe_embed_blob(blob)` | Extract embedding from BLOB | LIST(FLOAT) |
| `eupe_similarity(v1, v2)` | Cosine similarity | DOUBLE |
| `eupe_distance(v1, v2)` | L2 distance | DOUBLE |
| `eupe_normalize(vec)` | L2 normalize | LIST(FLOAT) |
| `eupe_dim(vec)` | Vector dimension | INTEGER |

### Model cache location

Downloaded models are cached at `~/.duckdb/extensions/eupe/models/<name>.onnx`.
Override with the `EUPE_MODEL_DIR` environment variable.

## EUPE Models

| Model | Params | Embed Dim |
|-------|--------|-----------|
| ViT-T/16 | 6M | 192 |
| ViT-S/16 | 21M | 384 |
| ViT-B/16 | 86M | 768 |
| ConvNeXt-T | 29M | varies |
| ConvNeXt-S | 50M | varies |
| ConvNeXt-B | 89M | varies |

## Use Cases

### Image Similarity Search
```sql
SELECT eupe_load_model('eupe_vit_s16.onnx');

CREATE TABLE image_embeddings AS
SELECT filename, eupe_embed(filepath) AS vec
FROM glob('/data/images/*.jpg');

-- Find most similar pairs
SELECT a.filename AS img_a, b.filename AS img_b,
       eupe_similarity(a.vec, b.vec) AS sim
FROM image_embeddings a, image_embeddings b
WHERE a.filename < b.filename
ORDER BY sim DESC LIMIT 10;
```

### Batch Embedding Extraction
```sql
SELECT filename,
       eupe_embed(filepath) AS embedding,
       eupe_dim(eupe_embed(filepath)) AS dim
FROM glob('/data/photos/*.png');
```

### Multi-Model Comparison
```sql
SELECT eupe_load_model('eupe_vit_t16.onnx', 'tiny');
SELECT eupe_load_model('eupe_vit_s16.onnx', 'small');

SELECT filename,
       eupe_embed(filepath, 'tiny') AS embed_tiny,
       eupe_embed(filepath, 'small') AS embed_small
FROM images;
```

## Preparing ONNX Models

Export EUPE models to ONNX format:

```python
import torch

model = torch.hub.load('path/to/EUPE', 'eupe_vits16', source='local',
                        weights='path/to/checkpoint.pth')
model.eval()

dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy, 'eupe_vit_s16.onnx',
                  input_names=['input'],
                  output_names=['embedding'],
                  dynamic_axes={'input': {0: 'batch'}})
```

## Embedding the tiny model into the extension binary

For the "just works after install" experience, bundle the ViT-T/16 model
directly into the extension binary. Export it (ideally FP16 for a ~12MB footprint),
then run:

```bash
./scripts/embed_tiny_model.sh path/to/eupe_vit_t16_fp16.onnx
make release
```

This regenerates `src/embedded/tiny_model_data.cpp` with the model bytes as a
C++ byte array. After rebuild, `eupe_embed()` works with no setup — the bundled
model is auto-loaded on first use. Without running the script, the stub remains
in place and users must load or download a model explicitly.

## Building

```bash
# Setup submodules
git submodule update --init --recursive

# Build
make release

# Test
make test
```

## Dependencies

- DuckDB (via submodule)
- ONNX Runtime (via vcpkg)
- stb_image.h (place in `src/include/` for PNG/JPEG support)

## License

MIT
