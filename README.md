# pic2vec - DuckDB Image Embedding Extension

`pic2vec` turns images into vectors directly from SQL. Built on ONNX Runtime,
bundled with Meta's **EUPE ViT-T/16** by default, works with any ONNX vision
model that takes NCHW input `[N, 3, H, W]`.

```sql
LOAD pic2vec;
SELECT pic2vec_embed('/photo.jpg');     -- → FLOAT[192] embedding (zero setup)
```

## Quick Start

### 1. Zero-setup (bundled EUPE ViT-T)

```sql
LOAD pic2vec;
SELECT pic2vec_embed('/path/to/photo.jpg');  -- bundled model auto-loads
SELECT pic2vec_bundled_info();               -- check what's bundled
```

### 2. Load an arbitrary ONNX model

```sql
SELECT pic2vec_load_model('/path/to/my_model.onnx', 'my_model');
SELECT pic2vec_embed('/path/to/photo.jpg', 'my_model');
```

### 3. Download a known model (whisper-style)

```sql
SELECT pic2vec_download_model('vit_s16');   -- → ~/.duckdb/extensions/pic2vec/models/
SELECT pic2vec_embed('/path/to/photo.jpg', 'vit_s16');
```

## Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `pic2vec_version()` | Extension version | VARCHAR |
| `pic2vec_bundled_info()` | Info about bundled model | VARCHAR |
| `pic2vec_load_bundled()` | Explicitly load bundled model | VARCHAR |
| `pic2vec_load_model(path[, name])` | Load ONNX model | VARCHAR |
| `pic2vec_download_model(name)` | Download known model + load | VARCHAR |
| `pic2vec_unload_model(name)` | Unload model | VARCHAR |
| `pic2vec_list_models()` | List loaded models | VARCHAR |
| `pic2vec_embed(image_path[, model])` | Image → embedding | LIST(FLOAT) |
| `pic2vec_embed_blob(blob)` | BLOB → embedding | LIST(FLOAT) |
| `pic2vec_similarity(v1, v2)` | Cosine similarity | DOUBLE |
| `pic2vec_distance(v1, v2)` | L2 distance | DOUBLE |
| `pic2vec_inner_product(v1, v2)` | Dot product | DOUBLE |
| `pic2vec_normalize(vec)` | L2 normalize | LIST(FLOAT) |
| `pic2vec_dim(vec)` | Vector dimension | INTEGER |

## When to use which: pic2vec vs built-in / vss

DuckDB already provides `array_cosine_similarity`, `array_distance`, and
`array_inner_product` for fixed-size `FLOAT[N]` arrays, plus the `vss`
extension for HNSW indexes. pic2vec deliberately keeps duplicate scalar
functions (`pic2vec_similarity`, etc.) so workflows that don't need HNSW
don't have to set up the `vss` extension or worry about dim-typed columns.

| Need | Recommended | Why |
|------|-------------|-----|
| Ad-hoc / one-shot similarity | **pic2vec_*** | No CAST, no schema, takes `LIST(FLOAT)` directly |
| Mixed-model embeddings in one query (192 + 384) | **pic2vec_*** | `LIST(FLOAT)` carries any dim; `FLOAT[N]` would force separate columns |
| Persisted table, ≤ 100k rows, full scan | Either | Both produce identical results |
| Persisted table, >> 100k rows, fast NN search | **`vss` + HNSW** | Built-in `array_cosine_distance` + index |
| Want to mix with Postgres-style array functions | **built-in `array_*`** | Works on `FLOAT[N]` natively |

## Composing with DuckDB built-ins

`pic2vec_embed()` returns `LIST(FLOAT)` — flexible (handles mixed model dims),
but DuckDB's built-in array distance functions and the HNSW index in `vss`
require fixed-size `FLOAT[N]`. Use a CAST to bridge them.

### Pattern A: Standalone (small datasets, ad-hoc queries)

Use pic2vec's own scalar functions. No CAST, no extension setup needed.

```sql
SELECT a, b, pic2vec_similarity(pic2vec_embed(a), pic2vec_embed(b)) AS sim
FROM (VALUES ('a.jpg', 'b.jpg'), ('c.jpg', 'd.jpg')) t(a, b);
```

### Pattern B: HNSW + vss (large datasets, fast NN search)

CAST the embedding to `FLOAT[N]` to enable HNSW indexing.

```sql
INSTALL vss; LOAD vss;

CREATE TABLE imgs (filename VARCHAR, vec FLOAT[192]);
INSERT INTO imgs
SELECT filename, pic2vec_embed(filename)::FLOAT[192]
FROM glob('/data/images/*.jpg');

CREATE INDEX vec_idx ON imgs USING HNSW (vec) WITH (metric = 'cosine');

-- Top-10 nearest to a query image
SELECT filename
FROM imgs
ORDER BY array_cosine_distance(vec, pic2vec_embed('/query.jpg')::FLOAT[192])
LIMIT 10;
```

### Pattern C: Mixed-dim models (multiple models in one query)

Keep `LIST(FLOAT)` for flexibility — different models have different dims.

```sql
SELECT pic2vec_load_model('/eupe_vit_t.onnx', 'tiny');
SELECT pic2vec_load_model('/eupe_vit_s.onnx', 'small');

CREATE TABLE imgs AS
SELECT filename,
       pic2vec_embed(filename, 'tiny')  AS vec_192,
       pic2vec_embed(filename, 'small') AS vec_384
FROM glob('/data/images/*.jpg');

-- Both columns use LIST(FLOAT); pic2vec_similarity handles either
SELECT pic2vec_similarity(a.vec_192, b.vec_192) AS sim_tiny,
       pic2vec_similarity(a.vec_384, b.vec_384) AS sim_small
FROM imgs a, imgs b WHERE a.filename < b.filename;
```

## Compatible models

Any ONNX vision model with NCHW input `[N, 3, H, W]` and a single embedding output:

- **EUPE** (ViT-T/S/B, ConvNeXt-T/S/B) — Meta
- **DINOv2 / DINOv3** — Meta
- **CLIP image encoder** — OpenAI
- **MobileNet, ResNet, EfficientNet, Swin, ConvNeXt** — torchvision/HuggingFace
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is
  applied internally
- Input H/W is read from ONNX metadata; preprocessing adapts

## Preparing ONNX Models

```python
import torch

model = torch.hub.load(...)  # your model
model.eval()

dummy = torch.randn(1, 3, 224, 224)  # adjust H/W to your model
torch.onnx.export(model, dummy, 'model.onnx',
                  input_names=['input'], output_names=['embedding'],
                  opset_version=14,
                  dynamic_axes={'input': {0: 'batch'}},
                  dynamo=False)  # legacy exporter is more permissive
```

## Embedding the bundled model into the extension binary

The extension repo ships a stub for the bundled tiny model. To produce a build
where `pic2vec_embed()` works zero-setup, run:

```bash
./scripts/embed_tiny_model.sh path/to/eupe_vit_t.onnx
make release
```

This regenerates `src/embedded/tiny_model_data.cpp` with the model bytes.

## Building

```bash
git submodule update --init --recursive
brew install onnxruntime    # or apt install / vcpkg
make release
```

## Dependencies

- DuckDB (via submodule, pinned to v1.4.1)
- ONNX Runtime (Homebrew/apt/vcpkg)
- stb_image.h in `src/include/` (for PNG/JPEG decoding)

## License

MIT
