#define DUCKDB_EXTENSION_MAIN

#include "eupe_extension.hpp"
#include "embedded_model.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"

#include <sstream>

namespace duckdb {

// ============================================
// eupe_version() -> VARCHAR
// ============================================
static void EupeVersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, "EUPE Extension v0.1.0 (Efficient Universal Perception Encoder)");
}

// ============================================
// eupe_load_model(model_path VARCHAR, model_name VARCHAR) -> VARCHAR
// ============================================
static void EupeLoadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &model_path_vec = args.data[0];
	auto &model_name_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat path_data;
	UnifiedVectorFormat name_data;
	model_path_vec.ToUnifiedFormat(count, path_data);
	model_name_vec.ToUnifiedFormat(count, name_data);

	auto result_data = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto path_idx = path_data.sel->get_index(i);
		auto name_idx = name_data.sel->get_index(i);

		if (!path_data.validity.RowIsValid(path_idx) || !name_data.validity.RowIsValid(name_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto model_path = ((string_t *)path_data.data)[path_idx].GetString();
		auto model_name = ((string_t *)name_data.data)[name_idx].GetString();

		auto &manager = EupeModelManager::Instance();
		auto msg = manager.LoadModel(model_path, model_name);
		result_data[i] = StringVector::AddString(result, msg);
	}
}

// ============================================
// eupe_load_model(model_path VARCHAR) -> VARCHAR
// (single arg: name derived from path)
// ============================================
static void EupeLoadModelSingleFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &model_path_vec = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat path_data;
	model_path_vec.ToUnifiedFormat(count, path_data);

	auto result_data = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto path_idx = path_data.sel->get_index(i);

		if (!path_data.validity.RowIsValid(path_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto model_path = ((string_t *)path_data.data)[path_idx].GetString();

		// Derive name from filename: /path/to/eupe_vit_s16.onnx -> eupe_vit_s16
		std::string model_name;
		auto slash_pos = model_path.find_last_of("/\\");
		auto dot_pos = model_path.find_last_of('.');
		if (slash_pos != std::string::npos) {
			model_name = model_path.substr(slash_pos + 1);
		} else {
			model_name = model_path;
		}
		if (dot_pos != std::string::npos && dot_pos > (slash_pos != std::string::npos ? slash_pos : 0)) {
			auto name_dot = model_name.find_last_of('.');
			if (name_dot != std::string::npos) {
				model_name = model_name.substr(0, name_dot);
			}
		}

		auto &manager = EupeModelManager::Instance();
		auto msg = manager.LoadModel(model_path, model_name);
		result_data[i] = StringVector::AddString(result, msg);
	}
}

// ============================================
// eupe_unload_model(model_name VARCHAR) -> VARCHAR
// ============================================
static void EupeUnloadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vec = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat name_data;
	name_vec.ToUnifiedFormat(count, name_data);

	auto result_data = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto idx = name_data.sel->get_index(i);
		if (!name_data.validity.RowIsValid(idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto model_name = ((string_t *)name_data.data)[idx].GetString();
		auto &manager = EupeModelManager::Instance();
		bool ok = manager.UnloadModel(model_name);
		if (ok) {
			result_data[i] = StringVector::AddString(result, "Unloaded model: " + model_name);
		} else {
			result_data[i] = StringVector::AddString(result, "Model not found: " + model_name);
		}
	}
}

// ============================================
// eupe_list_models() -> VARCHAR
// ============================================
static void EupeListModelsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);

	auto &manager = EupeModelManager::Instance();
	auto models = manager.ListModels();

	if (models.empty()) {
		result_data[0] = StringVector::AddString(result, "No models loaded. Use eupe_load_model() first.");
		return;
	}

	std::ostringstream oss;
	oss << "Loaded models (" << models.size() << "):";
	for (auto &name : models) {
		oss << "\n  - " << name;
	}
	result_data[0] = StringVector::AddString(result, oss.str());
}

// ============================================
// eupe_embed(image_path VARCHAR) -> LIST(FLOAT)
// Uses default model
// ============================================
static void EupeEmbedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &path_vec = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat path_data;
	path_vec.ToUnifiedFormat(count, path_data);

	auto &manager = EupeModelManager::Instance();
	auto *model = manager.EnsureDefaultModel();
	if (!model) {
		throw InvalidInputException(
		    "No EUPE model available. Call eupe_load_model(path), eupe_download_model(name), "
		    "or build with the bundled tiny model embedded (see scripts/embed_tiny_model.sh).");
	}

	auto &list_entry = ListVector::GetData(result);
	auto &child = ListVector::GetEntry(result);
	idx_t offset = 0;

	for (idx_t i = 0; i < count; i++) {
		auto idx = path_data.sel->get_index(i);

		if (!path_data.validity.RowIsValid(idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto image_path = ((string_t *)path_data.data)[idx].GetString();

		try {
			auto image = LoadAndPreprocessImage(image_path);
			auto embedding = RunInference(*model, image);

			list_entry[i].offset = offset;
			list_entry[i].length = embedding.size();

			idx_t new_size = offset + embedding.size();
			ListVector::Reserve(result, new_size);

			auto child_data = FlatVector::GetData<float>(child);
			for (idx_t j = 0; j < embedding.size(); j++) {
				child_data[offset + j] = embedding[j];
			}
			offset = new_size;
		} catch (std::exception &e) {
			throw IOException("Failed to process image '%s': %s", image_path, e.what());
		}
	}
	ListVector::SetListSize(result, offset);
}

// ============================================
// eupe_embed(image_path VARCHAR, model_name VARCHAR) -> LIST(FLOAT)
// ============================================
static void EupeEmbedWithModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &path_vec = args.data[0];
	auto &model_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat path_data;
	UnifiedVectorFormat model_data;
	path_vec.ToUnifiedFormat(count, path_data);
	model_vec.ToUnifiedFormat(count, model_data);

	auto &list_entry = ListVector::GetData(result);
	auto &child = ListVector::GetEntry(result);
	idx_t offset = 0;

	auto &manager = EupeModelManager::Instance();

	for (idx_t i = 0; i < count; i++) {
		auto p_idx = path_data.sel->get_index(i);
		auto m_idx = model_data.sel->get_index(i);

		if (!path_data.validity.RowIsValid(p_idx) || !model_data.validity.RowIsValid(m_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto image_path = ((string_t *)path_data.data)[p_idx].GetString();
		auto model_name = ((string_t *)model_data.data)[m_idx].GetString();

		auto *model = manager.GetModel(model_name);
		if (!model) {
			throw InvalidInputException("Model '%s' not loaded. Call eupe_load_model() first.", model_name);
		}

		try {
			auto image = LoadAndPreprocessImage(image_path);
			auto embedding = RunInference(*model, image);

			list_entry[i].offset = offset;
			list_entry[i].length = embedding.size();

			idx_t new_size = offset + embedding.size();
			ListVector::Reserve(result, new_size);

			auto child_data = FlatVector::GetData<float>(child);
			for (idx_t j = 0; j < embedding.size(); j++) {
				child_data[offset + j] = embedding[j];
			}
			offset = new_size;
		} catch (std::exception &e) {
			throw IOException("Failed to process image '%s': %s", image_path, e.what());
		}
	}
	ListVector::SetListSize(result, offset);
}

// ============================================
// eupe_embed_blob(image_blob BLOB) -> LIST(FLOAT)
// ============================================
static void EupeEmbedBlobFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &blob_vec = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat blob_data;
	blob_vec.ToUnifiedFormat(count, blob_data);

	auto &manager = EupeModelManager::Instance();
	auto *model = manager.EnsureDefaultModel();
	if (!model) {
		throw InvalidInputException(
		    "No EUPE model available. Call eupe_load_model(path), eupe_download_model(name), "
		    "or build with the bundled tiny model embedded (see scripts/embed_tiny_model.sh).");
	}

	auto &list_entry = ListVector::GetData(result);
	auto &child = ListVector::GetEntry(result);
	idx_t offset = 0;

	for (idx_t i = 0; i < count; i++) {
		auto idx = blob_data.sel->get_index(i);

		if (!blob_data.validity.RowIsValid(idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto blob = ((string_t *)blob_data.data)[idx];
		auto data_ptr = reinterpret_cast<const uint8_t *>(blob.GetData());
		auto data_size = blob.GetSize();

		try {
			auto image = LoadAndPreprocessImageFromMemory(data_ptr, data_size);
			auto embedding = RunInference(*model, image);

			list_entry[i].offset = offset;
			list_entry[i].length = embedding.size();

			idx_t new_size = offset + embedding.size();
			ListVector::Reserve(result, new_size);

			auto child_data = FlatVector::GetData<float>(child);
			for (idx_t j = 0; j < embedding.size(); j++) {
				child_data[offset + j] = embedding[j];
			}
			offset = new_size;
		} catch (std::exception &e) {
			throw IOException("Failed to process image blob: %s", e.what());
		}
	}
	ListVector::SetListSize(result, offset);
}

// ============================================
// Helper: extract float list from vector
// ============================================
static std::vector<float> ExtractFloatList(Vector &vec, idx_t row_idx) {
	auto list_data = ListVector::GetData(vec);
	auto &child = ListVector::GetEntry(vec);
	auto child_data = FlatVector::GetData<float>(child);

	auto entry = list_data[row_idx];
	std::vector<float> result(entry.length);
	for (idx_t j = 0; j < entry.length; j++) {
		result[j] = child_data[entry.offset + j];
	}
	return result;
}

// ============================================
// eupe_similarity(vec1 LIST(FLOAT), vec2 LIST(FLOAT)) -> DOUBLE
// ============================================
static void EupeSimilarityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &vec1 = args.data[0];
	auto &vec2 = args.data[1];
	idx_t count = args.size();

	auto result_data = FlatVector::GetData<double>(result);

	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(vec1, i) || FlatVector::IsNull(vec2, i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto a = ExtractFloatList(vec1, i);
		auto b = ExtractFloatList(vec2, i);

		if (a.size() != b.size()) {
			throw InvalidInputException(
			    "eupe_similarity: vector dimensions must match (%llu vs %llu)",
			    (unsigned long long)a.size(), (unsigned long long)b.size());
		}

		result_data[i] = CosineSimilarity(a, b);
	}
}

// ============================================
// eupe_distance(vec1 LIST(FLOAT), vec2 LIST(FLOAT)) -> DOUBLE
// ============================================
static void EupeDistanceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &vec1 = args.data[0];
	auto &vec2 = args.data[1];
	idx_t count = args.size();

	auto result_data = FlatVector::GetData<double>(result);

	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(vec1, i) || FlatVector::IsNull(vec2, i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto a = ExtractFloatList(vec1, i);
		auto b = ExtractFloatList(vec2, i);

		if (a.size() != b.size()) {
			throw InvalidInputException(
			    "eupe_distance: vector dimensions must match (%llu vs %llu)",
			    (unsigned long long)a.size(), (unsigned long long)b.size());
		}

		result_data[i] = L2Distance(a, b);
	}
}

// ============================================
// eupe_normalize(vec LIST(FLOAT)) -> LIST(FLOAT)
// L2 normalize an embedding vector
// ============================================
static void EupeNormalizeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vec = args.data[0];
	idx_t count = args.size();

	auto &list_entry = ListVector::GetData(result);
	auto &child = ListVector::GetEntry(result);
	idx_t offset = 0;

	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(input_vec, i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto vec = ExtractFloatList(input_vec, i);

		// L2 norm
		double norm = 0.0;
		for (auto v : vec) {
			norm += (double)v * v;
		}
		norm = std::sqrt(norm);

		list_entry[i].offset = offset;
		list_entry[i].length = vec.size();

		idx_t new_size = offset + vec.size();
		ListVector::Reserve(result, new_size);

		auto child_data = FlatVector::GetData<float>(child);
		if (norm > 1e-12) {
			for (idx_t j = 0; j < vec.size(); j++) {
				child_data[offset + j] = static_cast<float>(vec[j] / norm);
			}
		} else {
			for (idx_t j = 0; j < vec.size(); j++) {
				child_data[offset + j] = 0.0f;
			}
		}
		offset = new_size;
	}
	ListVector::SetListSize(result, offset);
}

// ============================================
// eupe_dim(vec LIST(FLOAT)) -> INTEGER
// ============================================
static void EupeDimFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vec = args.data[0];
	idx_t count = args.size();

	auto result_data = FlatVector::GetData<int32_t>(result);

	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(input_vec, i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto list_data = ListVector::GetData(input_vec);
		result_data[i] = static_cast<int32_t>(list_data[i].length);
	}
}

// ============================================
// eupe_download_model(model_name VARCHAR) -> VARCHAR
// ============================================
static void EupeDownloadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vec = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat name_data;
	name_vec.ToUnifiedFormat(count, name_data);

	auto result_data = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto idx = name_data.sel->get_index(i);
		if (!name_data.validity.RowIsValid(idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto model_name = ((string_t *)name_data.data)[idx].GetString();

		auto dl_msg = DownloadModel(model_name, /*force=*/false);
		auto cached_path = ResolveModelCachePath(model_name);
		auto load_msg = EupeModelManager::Instance().LoadModel(cached_path, model_name);

		std::string combined = dl_msg + "\n" + load_msg;
		result_data[i] = StringVector::AddString(result, combined);
	}
}

// ============================================
// eupe_load_bundled() -> VARCHAR
// ============================================
static void EupeLoadBundledFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	auto msg = EupeModelManager::Instance().LoadBundledTiny();
	result_data[0] = StringVector::AddString(result, msg);
}

// ============================================
// eupe_bundled_info() -> VARCHAR
// ============================================
static void EupeBundledInfoFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	std::ostringstream oss;
	if (EUPE_TINY_MODEL_SIZE > 0) {
		oss << "Bundled: " << EUPE_TINY_MODEL_DESC
		    << " (" << EUPE_TINY_MODEL_SIZE << " bytes)";
	} else {
		oss << "No model is bundled in this build. "
		       "Use eupe_download_model() or eupe_load_model().";
	}
	result_data[0] = StringVector::AddString(result, oss.str());
}

// ============================================
// Extension Load
// ============================================

void EupeExtension::Load(ExtensionLoader &loader) {
	// Version
	loader.RegisterFunction(ScalarFunction(
	    "eupe_version", {}, LogicalType::VARCHAR, EupeVersionFunction));

	// Model management
	loader.RegisterFunction(ScalarFunction(
	    "eupe_load_model",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::VARCHAR, EupeLoadModelFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_load_model",
	    {LogicalType::VARCHAR},
	    LogicalType::VARCHAR, EupeLoadModelSingleFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_unload_model",
	    {LogicalType::VARCHAR},
	    LogicalType::VARCHAR, EupeUnloadModelFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_list_models", {},
	    LogicalType::VARCHAR, EupeListModelsFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_download_model",
	    {LogicalType::VARCHAR},
	    LogicalType::VARCHAR, EupeDownloadModelFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_load_bundled", {},
	    LogicalType::VARCHAR, EupeLoadBundledFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_bundled_info", {},
	    LogicalType::VARCHAR, EupeBundledInfoFunction));

	// Embedding extraction
	loader.RegisterFunction(ScalarFunction(
	    "eupe_embed",
	    {LogicalType::VARCHAR},
	    LogicalType::LIST(LogicalType::FLOAT),
	    EupeEmbedFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_embed",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::LIST(LogicalType::FLOAT),
	    EupeEmbedWithModelFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_embed_blob",
	    {LogicalType::BLOB},
	    LogicalType::LIST(LogicalType::FLOAT),
	    EupeEmbedBlobFunction));

	// Vector operations
	loader.RegisterFunction(ScalarFunction(
	    "eupe_similarity",
	    {LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::DOUBLE, EupeSimilarityFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_distance",
	    {LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::DOUBLE, EupeDistanceFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_normalize",
	    {LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::LIST(LogicalType::FLOAT),
	    EupeNormalizeFunction));

	loader.RegisterFunction(ScalarFunction(
	    "eupe_dim",
	    {LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::INTEGER, EupeDimFunction));
}

std::string EupeExtension::Name() {
	return "eupe";
}

std::string EupeExtension::Version() const {
#ifdef EXT_VERSION_EUPE
	return EXT_VERSION_EUPE;
#else
	return "0.1.0";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void eupe_duckdb_cpp_init(duckdb::ExtensionLoader &loader) {
	duckdb::EupeExtension ext;
	ext.Load(loader);
}

DUCKDB_EXTENSION_API void eupe_init(duckdb::DatabaseInstance &db) {
	duckdb::Connection con(db);
	con.BeginTransaction();

	auto &catalog = duckdb::Catalog::GetSystemCatalog(*con.context);

	// Version
	duckdb::CreateScalarFunctionInfo version_func(duckdb::ScalarFunction(
	    "eupe_version", {}, duckdb::LogicalType::VARCHAR, duckdb::EupeVersionFunction));
	catalog.CreateFunction(*con.context, version_func);

	// Model management
	duckdb::CreateScalarFunctionInfo load_func(duckdb::ScalarFunction(
	    "eupe_load_model",
	    {duckdb::LogicalType::VARCHAR, duckdb::LogicalType::VARCHAR},
	    duckdb::LogicalType::VARCHAR, duckdb::EupeLoadModelFunction));
	catalog.CreateFunction(*con.context, load_func);

	duckdb::CreateScalarFunctionInfo load_single_func(duckdb::ScalarFunction(
	    "eupe_load_model",
	    {duckdb::LogicalType::VARCHAR},
	    duckdb::LogicalType::VARCHAR, duckdb::EupeLoadModelSingleFunction));
	catalog.CreateFunction(*con.context, load_single_func);

	duckdb::CreateScalarFunctionInfo unload_func(duckdb::ScalarFunction(
	    "eupe_unload_model",
	    {duckdb::LogicalType::VARCHAR},
	    duckdb::LogicalType::VARCHAR, duckdb::EupeUnloadModelFunction));
	catalog.CreateFunction(*con.context, unload_func);

	duckdb::CreateScalarFunctionInfo list_func(duckdb::ScalarFunction(
	    "eupe_list_models", {},
	    duckdb::LogicalType::VARCHAR, duckdb::EupeListModelsFunction));
	catalog.CreateFunction(*con.context, list_func);

	duckdb::CreateScalarFunctionInfo download_func(duckdb::ScalarFunction(
	    "eupe_download_model",
	    {duckdb::LogicalType::VARCHAR},
	    duckdb::LogicalType::VARCHAR, duckdb::EupeDownloadModelFunction));
	catalog.CreateFunction(*con.context, download_func);

	duckdb::CreateScalarFunctionInfo load_bundled_func(duckdb::ScalarFunction(
	    "eupe_load_bundled", {},
	    duckdb::LogicalType::VARCHAR, duckdb::EupeLoadBundledFunction));
	catalog.CreateFunction(*con.context, load_bundled_func);

	duckdb::CreateScalarFunctionInfo bundled_info_func(duckdb::ScalarFunction(
	    "eupe_bundled_info", {},
	    duckdb::LogicalType::VARCHAR, duckdb::EupeBundledInfoFunction));
	catalog.CreateFunction(*con.context, bundled_info_func);

	// Embedding
	duckdb::CreateScalarFunctionInfo embed_func(duckdb::ScalarFunction(
	    "eupe_embed",
	    {duckdb::LogicalType::VARCHAR},
	    duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT),
	    duckdb::EupeEmbedFunction));
	catalog.CreateFunction(*con.context, embed_func);

	duckdb::CreateScalarFunctionInfo embed_model_func(duckdb::ScalarFunction(
	    "eupe_embed",
	    {duckdb::LogicalType::VARCHAR, duckdb::LogicalType::VARCHAR},
	    duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT),
	    duckdb::EupeEmbedWithModelFunction));
	catalog.CreateFunction(*con.context, embed_model_func);

	duckdb::CreateScalarFunctionInfo embed_blob_func(duckdb::ScalarFunction(
	    "eupe_embed_blob",
	    {duckdb::LogicalType::BLOB},
	    duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT),
	    duckdb::EupeEmbedBlobFunction));
	catalog.CreateFunction(*con.context, embed_blob_func);

	// Vector operations
	duckdb::CreateScalarFunctionInfo sim_func(duckdb::ScalarFunction(
	    "eupe_similarity",
	    {duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT),
	     duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT)},
	    duckdb::LogicalType::DOUBLE, duckdb::EupeSimilarityFunction));
	catalog.CreateFunction(*con.context, sim_func);

	duckdb::CreateScalarFunctionInfo dist_func(duckdb::ScalarFunction(
	    "eupe_distance",
	    {duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT),
	     duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT)},
	    duckdb::LogicalType::DOUBLE, duckdb::EupeDistanceFunction));
	catalog.CreateFunction(*con.context, dist_func);

	duckdb::CreateScalarFunctionInfo norm_func(duckdb::ScalarFunction(
	    "eupe_normalize",
	    {duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT)},
	    duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT),
	    duckdb::EupeNormalizeFunction));
	catalog.CreateFunction(*con.context, norm_func);

	duckdb::CreateScalarFunctionInfo dim_func(duckdb::ScalarFunction(
	    "eupe_dim",
	    {duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT)},
	    duckdb::LogicalType::INTEGER, duckdb::EupeDimFunction));
	catalog.CreateFunction(*con.context, dim_func);

	con.Commit();
}

DUCKDB_EXTENSION_API const char *eupe_version() {
	return duckdb::DuckDB::LibraryVersion();
}

}
