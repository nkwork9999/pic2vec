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

struct EupeModel {
	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	std::string model_path;
	int64_t embed_dim;    // output embedding dimension
	int64_t input_height; // expected input height (default 256)
	int64_t input_width;  // expected input width  (default 256)
};

class EupeModelManager {
public:
	static EupeModelManager &Instance();

	// Load an ONNX model from path, returns status message
	std::string LoadModel(const std::string &model_path, const std::string &model_name);

	// Load an ONNX model from an in-memory byte buffer
	std::string LoadModelFromMemory(const uint8_t *data, size_t size, const std::string &model_name);

	// Load the bundled "tiny" model (ViT-T/16). Returns status message.
	// If the bundled model is not embedded in this build, returns an error.
	std::string LoadBundledTiny();

	// Ensure a default model is available, auto-loading the bundled tiny
	// model if nothing has been loaded yet. Returns nullptr if no model
	// is available (e.g. tiny was not embedded and user hasn't loaded one).
	EupeModel *EnsureDefaultModel();

	// Get a loaded model by name (returns nullptr if not found)
	EupeModel *GetModel(const std::string &model_name);

	// Get default model (first loaded, or nullptr)
	EupeModel *GetDefaultModel();

	// List loaded models
	std::vector<std::string> ListModels();

	// Unload a model
	bool UnloadModel(const std::string &model_name);

private:
	EupeModelManager() = default;
	std::mutex mutex;
	std::unordered_map<std::string, std::unique_ptr<EupeModel>> models;
	std::string default_model_name;
};

// ============================================
// Model download (whisper-style)
// ============================================

// Resolve the on-disk cache path for a downloaded model name.
// Uses $EUPE_MODEL_DIR if set, otherwise ~/.duckdb/extensions/eupe/models/
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

// Load image from file and preprocess for EUPE (256x256, ImageNet normalization)
ImageData LoadAndPreprocessImage(const std::string &file_path, int target_h = 256, int target_w = 256);

// Load image from raw bytes
ImageData LoadAndPreprocessImageFromMemory(const uint8_t *data, size_t size, int target_h = 256, int target_w = 256);

// Run ONNX inference, returns embedding vector
std::vector<float> RunInference(EupeModel &model, const ImageData &image);

// Cosine similarity between two vectors
double CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);

// L2 distance between two vectors
double L2Distance(const std::vector<float> &a, const std::vector<float> &b);

// ============================================
// Extension Class
// ============================================

class EupeExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb
