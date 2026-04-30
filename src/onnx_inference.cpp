#include "eupe_extension.hpp"
#include "embedded_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/types.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

namespace duckdb {

// ============================================
// EupeModelManager Singleton
// ============================================

EupeModelManager &EupeModelManager::Instance() {
	static EupeModelManager instance;
	return instance;
}

// Shared post-load step: probe input/output shapes and register the model.
// Assumes the caller already holds `mutex` and has constructed model->session.
static std::string FinalizeLoadedModel(EupeModelManager &manager,
                                       std::unordered_map<std::string, std::unique_ptr<EupeModel>> &models,
                                       std::string &default_model_name,
                                       std::unique_ptr<EupeModel> model,
                                       const std::string &model_name,
                                       const std::string &source_label) {
	// Probe output shape to determine embedding dimension
	auto output_info = model->session->GetOutputTypeInfo(0);
	auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
	auto output_shape = tensor_info.GetShape();

	if (output_shape.size() >= 2) {
		model->embed_dim = output_shape[1];
	} else if (output_shape.size() == 1) {
		model->embed_dim = output_shape[0];
	} else {
		model->embed_dim = 0;
	}

	// Probe input shape for H/W (expect NCHW)
	auto input_info = model->session->GetInputTypeInfo(0);
	auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
	auto input_shape = input_tensor_info.GetShape();
	if (input_shape.size() == 4) {
		if (input_shape[2] > 0) model->input_height = input_shape[2];
		if (input_shape[3] > 0) model->input_width = input_shape[3];
	}

	std::string info = "Loaded model '" + model_name + "' from " + source_label;
	info += " (embed_dim=" + std::to_string(model->embed_dim);
	info += ", input=" + std::to_string(model->input_height) + "x" + std::to_string(model->input_width) + ")";

	if (default_model_name.empty()) {
		default_model_name = model_name;
		info += " [default]";
	}

	models[model_name] = std::move(model);
	return info;
}

static Ort::SessionOptions MakeSessionOptions() {
	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(1);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	return opts;
}

std::string EupeModelManager::LoadModel(const std::string &model_path, const std::string &model_name) {
	std::lock_guard<std::mutex> lock(mutex);

	if (models.find(model_name) != models.end()) {
		return "Model '" + model_name + "' is already loaded.";
	}

	try {
		auto model = std::make_unique<EupeModel>();
		model->model_path = model_path;
		model->input_height = 256;
		model->input_width = 256;

		model->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, model_name.c_str());
		auto session_options = MakeSessionOptions();

#ifdef _WIN32
		std::wstring wide_path(model_path.begin(), model_path.end());
		model->session = std::make_unique<Ort::Session>(*model->env, wide_path.c_str(), session_options);
#else
		model->session = std::make_unique<Ort::Session>(*model->env, model_path.c_str(), session_options);
#endif

		return FinalizeLoadedModel(*this, models, default_model_name, std::move(model), model_name, model_path);

	} catch (const Ort::Exception &e) {
		return "Failed to load model '" + model_name + "': " + std::string(e.what());
	} catch (const std::exception &e) {
		return "Failed to load model '" + model_name + "': " + std::string(e.what());
	}
}

std::string EupeModelManager::LoadModelFromMemory(const uint8_t *data, size_t size, const std::string &model_name) {
	std::lock_guard<std::mutex> lock(mutex);

	if (models.find(model_name) != models.end()) {
		return "Model '" + model_name + "' is already loaded.";
	}

	if (!data || size == 0) {
		return "Failed to load model '" + model_name + "': no model bytes provided.";
	}

	try {
		auto model = std::make_unique<EupeModel>();
		model->model_path = "<embedded>";
		model->input_height = 256;
		model->input_width = 256;

		model->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, model_name.c_str());
		auto session_options = MakeSessionOptions();

		model->session = std::make_unique<Ort::Session>(*model->env, data, size, session_options);

		return FinalizeLoadedModel(*this, models, default_model_name, std::move(model), model_name,
		                           "<embedded " + std::to_string(size) + " bytes>");

	} catch (const Ort::Exception &e) {
		return "Failed to load model '" + model_name + "' from memory: " + std::string(e.what());
	} catch (const std::exception &e) {
		return "Failed to load model '" + model_name + "' from memory: " + std::string(e.what());
	}
}

std::string EupeModelManager::LoadBundledTiny() {
	if (EUPE_TINY_MODEL_SIZE == 0) {
		return "Bundled tiny model is not embedded in this build. "
		       "Run scripts/embed_tiny_model.sh <onnx> and rebuild, "
		       "or use eupe_download_model('vit_t16').";
	}
	return LoadModelFromMemory(EUPE_TINY_MODEL_DATA, EUPE_TINY_MODEL_SIZE, EUPE_TINY_MODEL_NAME);
}

EupeModel *EupeModelManager::EnsureDefaultModel() {
	{
		std::lock_guard<std::mutex> lock(mutex);
		if (!default_model_name.empty()) {
			auto it = models.find(default_model_name);
			if (it != models.end()) {
				return it->second.get();
			}
		}
	}
	// No default model yet — try auto-loading the bundled tiny model.
	if (EUPE_TINY_MODEL_SIZE > 0) {
		LoadBundledTiny();
		std::lock_guard<std::mutex> lock(mutex);
		auto it = models.find(EUPE_TINY_MODEL_NAME);
		if (it != models.end()) {
			return it->second.get();
		}
	}
	return nullptr;
}

EupeModel *EupeModelManager::GetModel(const std::string &model_name) {
	std::lock_guard<std::mutex> lock(mutex);
	auto it = models.find(model_name);
	if (it != models.end()) {
		return it->second.get();
	}
	return nullptr;
}

EupeModel *EupeModelManager::GetDefaultModel() {
	std::lock_guard<std::mutex> lock(mutex);
	if (default_model_name.empty()) {
		return nullptr;
	}
	auto it = models.find(default_model_name);
	if (it != models.end()) {
		return it->second.get();
	}
	return nullptr;
}

std::vector<std::string> EupeModelManager::ListModels() {
	std::lock_guard<std::mutex> lock(mutex);
	std::vector<std::string> names;
	names.reserve(models.size());
	for (auto &kv : models) {
		names.push_back(kv.first);
	}
	return names;
}

bool EupeModelManager::UnloadModel(const std::string &model_name) {
	std::lock_guard<std::mutex> lock(mutex);
	auto it = models.find(model_name);
	if (it == models.end()) {
		return false;
	}
	models.erase(it);
	if (default_model_name == model_name) {
		default_model_name = models.empty() ? "" : models.begin()->first;
	}
	return true;
}

// ============================================
// ONNX Inference
// ============================================

std::vector<float> RunInference(EupeModel &model, const ImageData &image) {
	if (!model.session) {
		throw std::runtime_error("Model session is not initialized");
	}

	// Prepare input tensor
	// EUPE expects NCHW format: [1, 3, H, W]
	std::array<int64_t, 4> input_shape = {1, 3, model.input_height, model.input_width};
	size_t input_size = 1 * 3 * model.input_height * model.input_width;

	if (image.pixels.size() != input_size) {
		throw std::runtime_error(
		    "Image pixel count mismatch: expected " + std::to_string(input_size) +
		    ", got " + std::to_string(image.pixels.size()));
	}

	// Create input tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto input_tensor = Ort::Value::CreateTensor<float>(
	    memory_info,
	    const_cast<float *>(image.pixels.data()),
	    input_size,
	    input_shape.data(),
	    input_shape.size());

	// Get input/output names
	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name = model.session->GetInputNameAllocated(0, allocator);
	auto output_name = model.session->GetOutputNameAllocated(0, allocator);

	const char *input_names[] = {input_name.get()};
	const char *output_names[] = {output_name.get()};

	// Run inference
	auto output_tensors = model.session->Run(
	    Ort::RunOptions{nullptr},
	    input_names, &input_tensor, 1,
	    output_names, 1);

	// Extract output
	auto &output_tensor = output_tensors[0];
	auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
	auto output_shape = output_info.GetShape();

	size_t output_size = 1;
	for (auto dim : output_shape) {
		if (dim > 0) {
			output_size *= dim;
		}
	}

	const float *output_data = output_tensor.GetTensorData<float>();
	std::vector<float> embedding(output_data, output_data + output_size);

	return embedding;
}

// ============================================
// Vector Operations
// ============================================

double CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
	if (a.size() != b.size() || a.empty()) {
		return 0.0;
	}

	double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
	for (size_t i = 0; i < a.size(); i++) {
		dot += (double)a[i] * b[i];
		norm_a += (double)a[i] * a[i];
		norm_b += (double)b[i] * b[i];
	}

	double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
	if (denom < 1e-12) {
		return 0.0;
	}
	return dot / denom;
}

double L2Distance(const std::vector<float> &a, const std::vector<float> &b) {
	if (a.size() != b.size() || a.empty()) {
		return 0.0;
	}

	double sum = 0.0;
	for (size_t i = 0; i < a.size(); i++) {
		double diff = (double)a[i] - b[i];
		sum += diff * diff;
	}
	return std::sqrt(sum);
}

// ============================================
// Model download (whisper-style)
// ============================================

// Known model catalog.
// These URLs are placeholders - replace with real HuggingFace / GitHub release URLs
// when the EUPE extension is published.
static const std::pair<const char *, const char *> KNOWN_MODELS[] = {
    {"vit_t16", "https://huggingface.co/eupe-onnx/eupe_vit_t16/resolve/main/eupe_vit_t16_fp16.onnx"},
    {"vit_s16", "https://huggingface.co/eupe-onnx/eupe_vit_s16/resolve/main/eupe_vit_s16_fp16.onnx"},
    {"vit_b16", "https://huggingface.co/eupe-onnx/eupe_vit_b16/resolve/main/eupe_vit_b16_fp16.onnx"},
    {"convnext_t", "https://huggingface.co/eupe-onnx/eupe_convnext_t/resolve/main/eupe_convnext_t_fp16.onnx"},
    {"convnext_s", "https://huggingface.co/eupe-onnx/eupe_convnext_s/resolve/main/eupe_convnext_s_fp16.onnx"},
    {"convnext_b", "https://huggingface.co/eupe-onnx/eupe_convnext_b/resolve/main/eupe_convnext_b_fp16.onnx"},
};

std::string LookupModelUrl(const std::string &model_name) {
	for (auto &entry : KNOWN_MODELS) {
		if (model_name == entry.first) {
			return entry.second;
		}
	}
	return "";
}

static bool FileExists(const std::string &path) {
	struct stat st;
	return stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFREG);
}

static bool DirExists(const std::string &path) {
	struct stat st;
	return stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFDIR);
}

static void EnsureDir(const std::string &path) {
	if (DirExists(path)) {
		return;
	}
	// Recursively create parent first
	auto slash = path.find_last_of("/\\");
	if (slash != std::string::npos && slash > 0) {
		EnsureDir(path.substr(0, slash));
	}
	MKDIR(path.c_str());
}

static std::string HomeDir() {
	const char *home = std::getenv("HOME");
#ifdef _WIN32
	if (!home) {
		home = std::getenv("USERPROFILE");
	}
#endif
	return home ? home : ".";
}

std::string ResolveModelCachePath(const std::string &model_name) {
	const char *override_dir = std::getenv("EUPE_MODEL_DIR");
	std::string dir;
	if (override_dir && *override_dir) {
		dir = override_dir;
	} else {
		dir = HomeDir() + "/.duckdb/extensions/eupe/models";
	}
	EnsureDir(dir);
	return dir + "/" + model_name + ".onnx";
}

std::string DownloadModel(const std::string &model_name, bool force) {
	auto url = LookupModelUrl(model_name);
	if (url.empty()) {
		return "Unknown model: '" + model_name + "'. Known: vit_t16, vit_s16, vit_b16, convnext_t, convnext_s, convnext_b.";
	}

	auto out_path = ResolveModelCachePath(model_name);

	if (!force && FileExists(out_path)) {
		return "Model '" + model_name + "' already cached at " + out_path + " (use force=true to redownload).";
	}

	// Use system curl for the download. This keeps us dependency-free at the
	// cost of requiring curl on the user's PATH (true on macOS, most Linux,
	// and Windows 10+).
	auto tmp_path = out_path + ".part";
	std::string cmd = "curl -fL --progress-bar -o '" + tmp_path + "' '" + url + "'";
	int rc = std::system(cmd.c_str());
	if (rc != 0) {
		std::remove(tmp_path.c_str());
		return "Download failed (curl exit " + std::to_string(rc) + "). URL: " + url;
	}
	std::rename(tmp_path.c_str(), out_path.c_str());
	return "Downloaded '" + model_name + "' to " + out_path;
}

} // namespace duckdb
