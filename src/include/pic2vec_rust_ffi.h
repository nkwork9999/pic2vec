#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, -1 on failure. On failure, call pic2vec_rs_last_error.
int32_t pic2vec_rs_load_model(const char *path, const char *name);
int32_t pic2vec_rs_load_model_from_memory(const uint8_t *data, size_t size, const char *name);

// Returns 1 if removed, 0 if not found.
int32_t pic2vec_rs_unload_model(const char *name);

// Returns a heap-allocated newline-separated UTF-8 string of model names.
// Caller MUST free with pic2vec_rs_free_string.
char *pic2vec_rs_list_models(void);

// Returns the embedding dim of the named model, or 0 if not loaded.
size_t pic2vec_rs_model_dim(const char *name);

// Returns a heap-allocated UTF-8 string with the last error message
// (empty string if no error). Caller MUST free with pic2vec_rs_free_string.
char *pic2vec_rs_last_error(void);

// Free a string returned by pic2vec_rs_list_models / pic2vec_rs_last_error.
void pic2vec_rs_free_string(char *p);

// Embed an image from a file path. On success writes a heap-allocated float
// array to *out_data and its length to *out_len. Caller MUST free with
// pic2vec_rs_free_floats. model_name may be NULL to use the default model.
int32_t pic2vec_rs_embed_path(const char *image_path,
                              const char *model_name,
                              float **out_data,
                              size_t *out_len);

// Embed an image from a byte buffer. Same ownership rules as embed_path.
int32_t pic2vec_rs_embed_blob(const uint8_t *data,
                              size_t size,
                              const char *model_name,
                              float **out_data,
                              size_t *out_len);

// Free a float array returned by pic2vec_rs_embed_*.
void pic2vec_rs_free_floats(float *ptr, size_t len);

// Probe an image's metadata. *out_format is heap-allocated; free with
// pic2vec_rs_free_string. Returns 0/-1.
int32_t pic2vec_rs_image_info(const char *path,
                              uint32_t *out_width,
                              uint32_t *out_height,
                              uint32_t *out_channels,
                              char **out_format);

// Compute a 64-bit dHash perceptual hash. Returns 0/-1.
int32_t pic2vec_rs_phash(const char *path, uint64_t *out_hash);

// Save annotated copies with red rectangles on patches whose cosine sim is
// below `threshold`. Returns the count of highlighted patches, or -1 on error.
int32_t pic2vec_rs_diff_save(const char *path_a,
                             const char *path_b,
                             const char *out_a,
                             const char *out_b,
                             const char *model_name,
                             double threshold);

// Save annotated copies with a red heatmap overlay (intensity ∝ 1 - sim).
// Returns 0/-1.
int32_t pic2vec_rs_diff_heatmap(const char *path_a,
                                const char *path_b,
                                const char *out_a,
                                const char *out_b,
                                const char *model_name);

// Patch-level diff. Requires the loaded model to have 2 outputs (cls, patches).
// Writes per-patch cosine similarities to *out_sims (heap, free with
// pic2vec_rs_free_floats), with patch count in *out_n.
int32_t pic2vec_rs_diff_patches(const char *path_a,
                                const char *path_b,
                                const char *model_name,
                                float **out_sims,
                                size_t *out_n);

// Batch embed: paths is an array of n null-terminated strings. On success
// writes a single contiguous f32 buffer of length n*per_embed (=*out_total_len)
// to *out_data, with each embedding's dim in *out_per_embed.
// Caller MUST free with pic2vec_rs_free_floats.
int32_t pic2vec_rs_embed_batch(const char *const *paths,
                               size_t n_paths,
                               const char *model_name,
                               float **out_data,
                               size_t *out_total_len,
                               size_t *out_per_embed);

#ifdef __cplusplus
}
#endif
