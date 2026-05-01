#pragma once

#include "duckdb/main/extension.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declaration for ONNX Runtime
namespace Ort {
class Session;
class Env;
} // namespace Ort

namespace duckdb {

// ============================================
// ONNX Inference Engine
// ============================================

struct Pic2VecModel {
	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	std::string model_path;
	int64_t embed_dim;    // output embedding dimension
	int64_t input_height; // expected input height (default 256)
	int64_t input_width;  // expected input width  (default 256)
};

class Pic2VecModelManager {
public:
	static Pic2VecModelManager &Instance();

	// Load an ONNX model from path, returns status message
	std::string LoadModel(const std::string &model_path, const std::string &model_name);

	// Load an ONNX model from an in-memory byte buffer
	std::string LoadModelFromMemory(const uint8_t *data, size_t size, const std::string &model_name);

	// Load the bundled tiny model (currently EUPE ViT-T/16). Returns status message.
	// If no model is embedded in this build, returns an error.
	std::string LoadBundledTiny();

	// Ensure a default model is available, auto-loading the bundled tiny
	// model if nothing has been loaded yet. Returns nullptr if no model
	// is available (e.g. tiny was not embedded and user hasn't loaded one).
	Pic2VecModel *EnsureDefaultModel();

	// Get a loaded model by name (returns nullptr if not found)
	Pic2VecModel *GetModel(const std::string &model_name);

	// Get default model (first loaded, or nullptr)
	Pic2VecModel *GetDefaultModel();

	// List loaded models
	std::vector<std::string> ListModels();

	// Unload a model
	bool UnloadModel(const std::string &model_name);

private:
	Pic2VecModelManager() = default;
	std::mutex mutex;
	std::unordered_map<std::string, std::unique_ptr<Pic2VecModel>> models;
	std::string default_model_name;
};

// ============================================
// Model download (whisper-style)
// ============================================

// Resolve the on-disk cache path for a downloaded model name.
// Uses $PIC2VEC_MODEL_DIR if set, otherwise ~/.duckdb/extensions/pic2vec/models/
std::string ResolveModelCachePath(const std::string &model_name);

// Known model catalog: maps short names (e.g. "vit_s16") to download URLs.
// Returns empty string if name is not in the catalog.
std::string LookupModelUrl(const std::string &model_name);

// Download a model to the local cache. Returns status message.
// Uses system `curl` for transport; no download is performed if the
// file already exists in the cache (unless force is true).
std::string DownloadModel(const std::string &model_name, bool force = false);

// ============================================
// Image Utilities
// ============================================

struct ImageData {
	std::vector<float> pixels; // CHW format, normalized
	int width;
	int height;
	int channels;
};

// Load image from file and preprocess (resize + ImageNet normalization)
ImageData LoadAndPreprocessImage(const std::string &file_path, int target_h = 256, int target_w = 256);

// Load image from raw bytes
ImageData LoadAndPreprocessImageFromMemory(const uint8_t *data, size_t size, int target_h = 256, int target_w = 256);

// Run ONNX inference, returns embedding vector
std::vector<float> RunInference(Pic2VecModel &model, const ImageData &image);

// Cosine similarity between two vectors
double CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);

// L2 distance between two vectors
double L2Distance(const std::vector<float> &a, const std::vector<float> &b);

// Inner (dot) product between two vectors
double InnerProduct(const std::vector<float> &a, const std::vector<float> &b);

// ============================================
// Extension Class
// ============================================

class Pic2vecExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb
