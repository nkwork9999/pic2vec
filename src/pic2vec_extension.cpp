#define DUCKDB_EXTENSION_MAIN

#include "pic2vec_extension.hpp"
#include "pic2vec_rust_ffi.h"
#include "embedded_model.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/types.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

namespace duckdb {

// ============================================
// Vector operations (pure C++)
// ============================================

double CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
	if (a.size() != b.size() || a.empty()) return 0.0;
	double dot = 0, na = 0, nb = 0;
	for (size_t i = 0; i < a.size(); i++) {
		dot += (double)a[i] * b[i];
		na += (double)a[i] * a[i];
		nb += (double)b[i] * b[i];
	}
	double denom = std::sqrt(na) * std::sqrt(nb);
	return denom < 1e-12 ? 0.0 : dot / denom;
}

double L2Distance(const std::vector<float> &a, const std::vector<float> &b) {
	if (a.size() != b.size() || a.empty()) return 0.0;
	double sum = 0;
	for (size_t i = 0; i < a.size(); i++) {
		double d = (double)a[i] - b[i];
		sum += d * d;
	}
	return std::sqrt(sum);
}

double InnerProduct(const std::vector<float> &a, const std::vector<float> &b) {
	if (a.size() != b.size() || a.empty()) return 0.0;
	double dot = 0;
	for (size_t i = 0; i < a.size(); i++) dot += (double)a[i] * b[i];
	return dot;
}

// ============================================
// Bundled tiny model auto-load (one-shot)
// ============================================

static std::once_flag bundled_load_flag;
static bool bundled_load_attempted_ok = false;

static void EnsureBundledLoaded() {
	std::call_once(bundled_load_flag, []() {
		if (EUPE_TINY_MODEL_SIZE == 0) return;
		auto rc = pic2vec_rs_load_model_from_memory(
		    EUPE_TINY_MODEL_DATA, EUPE_TINY_MODEL_SIZE, EUPE_TINY_MODEL_NAME);
		bundled_load_attempted_ok = (rc == 0);
	});
}

static std::string LastRustError() {
	auto *p = pic2vec_rs_last_error();
	if (!p) return "";
	std::string s(p);
	pic2vec_rs_free_string(p);
	return s;
}

// ============================================
// Helpers
// ============================================

// Flatten the LIST vector so we can index uniformly regardless of how the
// upstream operator (CREATE TABLE AS, ORDER BY, etc.) chose to encode it.
// Without this, dictionary/constant-encoded list inputs return length 0.
static std::vector<float> ExtractFloatList(Vector &vec, idx_t row_idx, idx_t count) {
	vec.Flatten(count);
	auto list_data = ListVector::GetData(vec);
	auto &child = ListVector::GetEntry(vec);
	auto child_data = FlatVector::GetData<float>(child);
	auto entry = list_data[row_idx];
	std::vector<float> out(entry.length);
	for (idx_t j = 0; j < entry.length; j++) {
		out[j] = child_data[entry.offset + j];
	}
	return out;
}

static void WriteEmbedding(Vector &result, idx_t row, const float *data, size_t len, idx_t &offset) {
	auto list_entry = ListVector::GetData(result);
	list_entry[row].offset = offset;
	list_entry[row].length = len;

	idx_t new_size = offset + len;
	ListVector::Reserve(result, new_size);

	auto &child = ListVector::GetEntry(result);
	auto child_data = FlatVector::GetData<float>(child);
	for (size_t j = 0; j < len; j++) child_data[offset + j] = data[j];
	offset = new_size;
}

// ============================================
// SQL: pic2vec_version() -> VARCHAR
// ============================================
static void Pic2VecVersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto rd = ConstantVector::GetData<string_t>(result);
	rd[0] = StringVector::AddString(result,
	    "pic2vec v0.1.0 - DuckDB image embedding (tract-onnx, bundles EUPE ViT-T)");
}

// ============================================
// SQL: pic2vec_bundled_info() -> VARCHAR
// ============================================
static void Pic2VecBundledInfoFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto rd = ConstantVector::GetData<string_t>(result);
	std::ostringstream oss;
	if (EUPE_TINY_MODEL_SIZE > 0) {
		oss << "Bundled: " << EUPE_TINY_MODEL_DESC << " (" << EUPE_TINY_MODEL_SIZE << " bytes)";
	} else {
		oss << "No model is bundled in this build. "
		       "Use pic2vec_download_model() or pic2vec_load_model().";
	}
	rd[0] = StringVector::AddString(result, oss.str());
}

// ============================================
// SQL: pic2vec_load_bundled() -> VARCHAR
// ============================================
static void Pic2VecLoadBundledFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto rd = ConstantVector::GetData<string_t>(result);
	if (EUPE_TINY_MODEL_SIZE == 0) {
		rd[0] = StringVector::AddString(result,
		    "Bundled tiny model is not embedded in this build. "
		    "Run scripts/embed_tiny_model.sh <onnx> and rebuild.");
		return;
	}
	auto rc = pic2vec_rs_load_model_from_memory(
	    EUPE_TINY_MODEL_DATA, EUPE_TINY_MODEL_SIZE, EUPE_TINY_MODEL_NAME);
	if (rc == 0) {
		bundled_load_attempted_ok = true;
		rd[0] = StringVector::AddString(result,
		    std::string("Loaded bundled '") + EUPE_TINY_MODEL_NAME + "': " + EUPE_TINY_MODEL_DESC);
	} else {
		rd[0] = StringVector::AddString(result,
		    std::string("Failed to load bundled tiny: ") + LastRustError());
	}
}

// ============================================
// SQL: pic2vec_load_model(path[, name]) -> VARCHAR
// ============================================
static void DoLoadModel(const std::string &path, const std::string &name, std::string &msg) {
	auto rc = pic2vec_rs_load_model(path.c_str(), name.c_str());
	if (rc == 0) {
		size_t dim = pic2vec_rs_model_dim(name.c_str());
		msg = "Loaded model '" + name + "' from " + path + " (embed_dim=" + std::to_string(dim) + ")";
	} else {
		msg = "Failed to load '" + name + "': " + LastRustError();
	}
}

static std::string DeriveNameFromPath(const std::string &p) {
	auto slash = p.find_last_of("/\\");
	std::string base = (slash != std::string::npos) ? p.substr(slash + 1) : p;
	auto dot = base.find_last_of('.');
	if (dot != std::string::npos) base = base.substr(0, dot);
	return base.empty() ? "model" : base;
}

static void Pic2VecLoadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat pf, nf;
	args.data[0].ToUnifiedFormat(count, pf);
	args.data[1].ToUnifiedFormat(count, nf);
	auto rd = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto pi = pf.sel->get_index(i), ni = nf.sel->get_index(i);
		if (!pf.validity.RowIsValid(pi) || !nf.validity.RowIsValid(ni)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto path = ((string_t *)pf.data)[pi].GetString();
		auto name = ((string_t *)nf.data)[ni].GetString();
		std::string msg;
		DoLoadModel(path, name, msg);
		rd[i] = StringVector::AddString(result, msg);
	}
}

static void Pic2VecLoadModelSingleFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat pf;
	args.data[0].ToUnifiedFormat(count, pf);
	auto rd = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto pi = pf.sel->get_index(i);
		if (!pf.validity.RowIsValid(pi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto path = ((string_t *)pf.data)[pi].GetString();
		auto name = DeriveNameFromPath(path);
		std::string msg;
		DoLoadModel(path, name, msg);
		rd[i] = StringVector::AddString(result, msg);
	}
}

// ============================================
// SQL: pic2vec_unload_model(name) -> VARCHAR
// ============================================
static void Pic2VecUnloadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat nf;
	args.data[0].ToUnifiedFormat(count, nf);
	auto rd = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto ni = nf.sel->get_index(i);
		if (!nf.validity.RowIsValid(ni)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto name = ((string_t *)nf.data)[ni].GetString();
		auto rc = pic2vec_rs_unload_model(name.c_str());
		rd[i] = StringVector::AddString(result,
		    rc == 1 ? ("Unloaded model: " + name) : ("Model not found: " + name));
	}
}

// ============================================
// SQL: pic2vec_list_models() -> VARCHAR
// ============================================
static void Pic2VecListModelsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto rd = ConstantVector::GetData<string_t>(result);

	auto *p = pic2vec_rs_list_models();
	std::string list = p ? std::string(p) : "";
	if (p) pic2vec_rs_free_string(p);

	if (list.empty()) {
		rd[0] = StringVector::AddString(result, "No models loaded. Use pic2vec_load_model() first.");
		return;
	}

	std::ostringstream oss;
	int n = 1;
	for (size_t i = 0; i < list.size(); i++) if (list[i] == '\n') n++;
	oss << "Loaded models (" << n << "):";
	std::istringstream is(list);
	std::string line;
	while (std::getline(is, line)) oss << "\n  - " << line;
	rd[0] = StringVector::AddString(result, oss.str());
}

// ============================================
// SQL: pic2vec_embed(path[, model]) -> LIST(FLOAT)
// ============================================
static void Pic2VecEmbedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	EnsureBundledLoaded();
	auto count = args.size();
	UnifiedVectorFormat pf;
	args.data[0].ToUnifiedFormat(count, pf);
	idx_t offset = 0;

	for (idx_t i = 0; i < count; i++) {
		auto pi = pf.sel->get_index(i);
		if (!pf.validity.RowIsValid(pi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto path = ((string_t *)pf.data)[pi].GetString();

		float *out_data = nullptr;
		size_t out_len = 0;
		auto rc = pic2vec_rs_embed_path(path.c_str(), nullptr, &out_data, &out_len);
		if (rc != 0) {
			throw IOException("pic_embed: %s", LastRustError().c_str());
		}
		WriteEmbedding(result, i, out_data, out_len, offset);
		pic2vec_rs_free_floats(out_data, out_len);
	}
	ListVector::SetListSize(result, offset);
}

static void Pic2VecEmbedWithModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat pf, mf;
	args.data[0].ToUnifiedFormat(count, pf);
	args.data[1].ToUnifiedFormat(count, mf);
	idx_t offset = 0;

	for (idx_t i = 0; i < count; i++) {
		auto pi = pf.sel->get_index(i), mi = mf.sel->get_index(i);
		if (!pf.validity.RowIsValid(pi) || !mf.validity.RowIsValid(mi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto path = ((string_t *)pf.data)[pi].GetString();
		auto model = ((string_t *)mf.data)[mi].GetString();

		float *out_data = nullptr;
		size_t out_len = 0;
		auto rc = pic2vec_rs_embed_path(path.c_str(), model.c_str(), &out_data, &out_len);
		if (rc != 0) {
			throw IOException("pic_embed: %s", LastRustError().c_str());
		}
		WriteEmbedding(result, i, out_data, out_len, offset);
		pic2vec_rs_free_floats(out_data, out_len);
	}
	ListVector::SetListSize(result, offset);
}

static void Pic2VecEmbedBlobFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	EnsureBundledLoaded();
	auto count = args.size();
	UnifiedVectorFormat bf;
	args.data[0].ToUnifiedFormat(count, bf);
	idx_t offset = 0;

	for (idx_t i = 0; i < count; i++) {
		auto bi = bf.sel->get_index(i);
		if (!bf.validity.RowIsValid(bi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto blob = ((string_t *)bf.data)[bi];

		float *out_data = nullptr;
		size_t out_len = 0;
		auto rc = pic2vec_rs_embed_blob(
		    reinterpret_cast<const uint8_t *>(blob.GetData()), blob.GetSize(),
		    nullptr, &out_data, &out_len);
		if (rc != 0) {
			throw IOException("pic_embed_blob: %s", LastRustError().c_str());
		}
		WriteEmbedding(result, i, out_data, out_len, offset);
		pic2vec_rs_free_floats(out_data, out_len);
	}
	ListVector::SetListSize(result, offset);
}

// ============================================
// SQL: vector ops (pure C++)
// ============================================
static void Pic2VecSimilarityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<double>(result);
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(args.data[0], i) || FlatVector::IsNull(args.data[1], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto a = ExtractFloatList(args.data[0], i, count);
		auto b = ExtractFloatList(args.data[1], i, count);
		if (a.size() != b.size()) {
			throw InvalidInputException("pic_similarity: dims must match (%llu vs %llu)",
			    (unsigned long long)a.size(), (unsigned long long)b.size());
		}
		rd[i] = CosineSimilarity(a, b);
	}
}

static void Pic2VecDistanceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<double>(result);
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(args.data[0], i) || FlatVector::IsNull(args.data[1], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto a = ExtractFloatList(args.data[0], i, count);
		auto b = ExtractFloatList(args.data[1], i, count);
		if (a.size() != b.size()) {
			throw InvalidInputException("pic_distance: dims must match");
		}
		rd[i] = L2Distance(a, b);
	}
}

static void Pic2VecInnerProductFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<double>(result);
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(args.data[0], i) || FlatVector::IsNull(args.data[1], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto a = ExtractFloatList(args.data[0], i, count);
		auto b = ExtractFloatList(args.data[1], i, count);
		if (a.size() != b.size()) {
			throw InvalidInputException("pic_inner_product: dims must match");
		}
		rd[i] = InnerProduct(a, b);
	}
}

// ============================================
// SQL: pic_image_info(path) -> STRUCT(width INTEGER, height INTEGER, channels INTEGER, format VARCHAR)
// ============================================
static void Pic2VecImageInfoFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat pf;
	args.data[0].ToUnifiedFormat(count, pf);

	auto &children = StructVector::GetEntries(result);
	auto width_data = FlatVector::GetData<int32_t>(*children[0]);
	auto height_data = FlatVector::GetData<int32_t>(*children[1]);
	auto channels_data = FlatVector::GetData<int32_t>(*children[2]);

	for (idx_t i = 0; i < count; i++) {
		auto pi = pf.sel->get_index(i);
		if (!pf.validity.RowIsValid(pi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto path = ((string_t *)pf.data)[pi].GetString();
		uint32_t w = 0, h = 0, ch = 0;
		char *fmt = nullptr;
		auto rc = pic2vec_rs_image_info(path.c_str(), &w, &h, &ch, &fmt);
		if (rc != 0) {
			throw IOException("pic_image_info: %s", LastRustError().c_str());
		}
		width_data[i] = (int32_t)w;
		height_data[i] = (int32_t)h;
		channels_data[i] = (int32_t)ch;
		FlatVector::GetData<string_t>(*children[3])[i] =
		    StringVector::AddString(*children[3], fmt ? fmt : "unknown");
		if (fmt) pic2vec_rs_free_string(fmt);
	}
}

// ============================================
// SQL: pic_phash(path) -> BIGINT (64-bit dHash)
// ============================================
static void Pic2VecPhashFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat pf;
	args.data[0].ToUnifiedFormat(count, pf);
	auto rd = FlatVector::GetData<int64_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto pi = pf.sel->get_index(i);
		if (!pf.validity.RowIsValid(pi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto path = ((string_t *)pf.data)[pi].GetString();
		uint64_t hash = 0;
		auto rc = pic2vec_rs_phash(path.c_str(), &hash);
		if (rc != 0) {
			throw IOException("pic_phash: %s", LastRustError().c_str());
		}
		rd[i] = (int64_t)hash;
	}
}

// ============================================
// SQL: pic_hamming(a BIGINT, b BIGINT) -> INTEGER
// Hamming distance between two 64-bit phashes.
// ============================================
static void Pic2VecHammingFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat af, bf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);
	auto adata = (int64_t *)af.data;
	auto bdata = (int64_t *)bf.data;
	auto rd = FlatVector::GetData<int32_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		uint64_t x = (uint64_t)adata[ai] ^ (uint64_t)bdata[bi];
		rd[i] = __builtin_popcountll(x);
	}
}

// ============================================
// SQL: pic_diff_patches(path_a, path_b) -> LIST(STRUCT(row, col, sim))
// Per-patch cosine similarity between two images. Lower sim = bigger spatial
// difference. Patch grid is sqrt(N)*sqrt(N) (e.g. 14x14 for ViT-T/16 at 224).
// ============================================
static void Pic2VecDiffPatchesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	EnsureBundledLoaded();
	auto count = args.size();
	UnifiedVectorFormat af, bf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);

	auto out_list_data = ListVector::GetData(result);
	idx_t offset = 0;

	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto pa = ((string_t *)af.data)[ai].GetString();
		auto pb = ((string_t *)bf.data)[bi].GetString();

		float *sims = nullptr;
		size_t n = 0;
		auto rc = pic2vec_rs_diff_patches(pa.c_str(), pb.c_str(), nullptr, &sims, &n);
		if (rc != 0) {
			throw IOException("pic_diff_patches: %s", LastRustError().c_str());
		}

		// Determine grid side (ViT patches are square: 14x14 for 196 patches).
		size_t grid_side = (size_t)std::sqrt((double)n + 0.5);
		if (grid_side * grid_side != n) {
			pic2vec_rs_free_floats(sims, n);
			throw InvalidInputException(
			    "pic_diff_patches: non-square patch grid (%llu)", (unsigned long long)n);
		}

		out_list_data[i].offset = offset;
		out_list_data[i].length = n;
		ListVector::Reserve(result, offset + n);

		auto &child = ListVector::GetEntry(result);    // STRUCT
		auto &struct_children = StructVector::GetEntries(child);
		auto row_data = FlatVector::GetData<int32_t>(*struct_children[0]);
		auto col_data = FlatVector::GetData<int32_t>(*struct_children[1]);
		auto sim_data = FlatVector::GetData<double>(*struct_children[2]);
		for (size_t k = 0; k < n; k++) {
			row_data[offset + k] = (int32_t)(k / grid_side);
			col_data[offset + k] = (int32_t)(k % grid_side);
			sim_data[offset + k] = (double)sims[k];
		}
		offset += n;
		pic2vec_rs_free_floats(sims, n);
	}
	ListVector::SetListSize(result, offset);
}

// ============================================
// SQL: pic_diff_save(a, b, out_a, out_b[, threshold]) -> INTEGER (count)
// Save annotated copies of a/b with red rectangles around the patches
// whose cosine similarity falls below `threshold`. Default 0.95.
// ============================================
static void Pic2VecDiffSaveImpl(DataChunk &args, Vector &result, bool has_threshold) {
	EnsureBundledLoaded();
	auto count = args.size();
	UnifiedVectorFormat af, bf, oaf, obf, tf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);
	args.data[2].ToUnifiedFormat(count, oaf);
	args.data[3].ToUnifiedFormat(count, obf);
	if (has_threshold) args.data[4].ToUnifiedFormat(count, tf);

	auto rd = FlatVector::GetData<int32_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		auto oai = oaf.sel->get_index(i), obi = obf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi) ||
		    !oaf.validity.RowIsValid(oai) || !obf.validity.RowIsValid(obi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto pa = ((string_t *)af.data)[ai].GetString();
		auto pb = ((string_t *)bf.data)[bi].GetString();
		auto oa = ((string_t *)oaf.data)[oai].GetString();
		auto ob = ((string_t *)obf.data)[obi].GetString();
		double threshold = has_threshold
		    ? ((double *)tf.data)[tf.sel->get_index(i)]
		    : 0.95;
		auto rc = pic2vec_rs_diff_save(pa.c_str(), pb.c_str(),
		                               oa.c_str(), ob.c_str(),
		                               nullptr, threshold);
		if (rc < 0) {
			throw IOException("pic_diff_save: %s", LastRustError().c_str());
		}
		rd[i] = rc;
	}
}

static void Pic2VecDiffSaveFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	Pic2VecDiffSaveImpl(args, result, false);
}
static void Pic2VecDiffSaveThresholdFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	Pic2VecDiffSaveImpl(args, result, true);
}

// ============================================
// SQL: pic_diff_heatmap(a, b, out_a, out_b) -> BOOLEAN (success)
// ============================================
static void Pic2VecDiffHeatmapFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	EnsureBundledLoaded();
	auto count = args.size();
	UnifiedVectorFormat af, bf, oaf, obf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);
	args.data[2].ToUnifiedFormat(count, oaf);
	args.data[3].ToUnifiedFormat(count, obf);

	auto rd = FlatVector::GetData<bool>(result);
	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		auto oai = oaf.sel->get_index(i), obi = obf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi) ||
		    !oaf.validity.RowIsValid(oai) || !obf.validity.RowIsValid(obi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto pa = ((string_t *)af.data)[ai].GetString();
		auto pb = ((string_t *)bf.data)[bi].GetString();
		auto oa = ((string_t *)oaf.data)[oai].GetString();
		auto ob = ((string_t *)obf.data)[obi].GetString();
		auto rc = pic2vec_rs_diff_heatmap(pa.c_str(), pb.c_str(),
		                                  oa.c_str(), ob.c_str(), nullptr);
		if (rc != 0) {
			throw IOException("pic_diff_heatmap: %s", LastRustError().c_str());
		}
		rd[i] = true;
	}
}

// ============================================
// SQL: pic_diff_ascii(path_a, path_b[, threshold]) -> VARCHAR
// CLI-friendly visualization of patch-level differences. Cells below
// `threshold` (default 0.95) are marked '*'; matching cells are '.'.
// ============================================
static constexpr double DEFAULT_DIFF_ASCII_THRESHOLD = 0.95;

static void Pic2VecDiffAsciiImpl(DataChunk &args, Vector &result, bool has_threshold) {
	EnsureBundledLoaded();
	auto count = args.size();
	UnifiedVectorFormat af, bf, tf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);
	if (has_threshold) args.data[2].ToUnifiedFormat(count, tf);

	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto pa = ((string_t *)af.data)[ai].GetString();
		auto pb = ((string_t *)bf.data)[bi].GetString();
		double threshold = has_threshold
		    ? ((double *)tf.data)[tf.sel->get_index(i)]
		    : DEFAULT_DIFF_ASCII_THRESHOLD;

		float *sims = nullptr;
		size_t n = 0;
		auto rc = pic2vec_rs_diff_patches(pa.c_str(), pb.c_str(), nullptr, &sims, &n);
		if (rc != 0) {
			throw IOException("pic_diff_ascii: %s", LastRustError().c_str());
		}
		size_t side = (size_t)std::sqrt((double)n + 0.5);
		if (side * side != n) {
			pic2vec_rs_free_floats(sims, n);
			throw InvalidInputException("pic_diff_ascii: non-square patch grid (%llu)",
			    (unsigned long long)n);
		}

		std::ostringstream out;
		// Header: column numbers (0..side-1)
		out << "       ";
		for (size_t c = 0; c < side; c++) {
			if (c < 10) out << c << " ";
			else out << c;
		}
		out << "\n";

		// Find the single most-different patch for the arrow marker.
		size_t min_idx = 0;
		float min_val = sims[0];
		for (size_t k = 1; k < n; k++) {
			if (sims[k] < min_val) {
				min_val = sims[k];
				min_idx = k;
			}
		}

		for (size_t r = 0; r < side; r++) {
			bool is_min_row = (r == min_idx / side);
			out << (is_min_row ? " ->  " : "     ");
			out << (r < 10 ? " " : "") << r << " |";
			for (size_t c = 0; c < side; c++) {
				bool is_diff = sims[r * side + c] < threshold;
				out << (is_diff ? "* " : ". ");
			}
			out << "|";
			// Annotate first row index with min sim col(s)
			bool first = true;
			for (size_t c = 0; c < side; c++) {
				if (sims[r * side + c] < threshold) {
					if (first) {
						out << "  row=" << r << ": col=" << c;
						first = false;
					} else {
						out << "," << c;
					}
				}
			}
			if (!first && is_min_row) {
				out << "   <- min sim=" << std::fixed << std::setprecision(3) << sims[min_idx];
			}
			out << "\n";
		}

		FlatVector::GetData<string_t>(result)[i] =
		    StringVector::AddString(result, out.str());
		pic2vec_rs_free_floats(sims, n);
	}
}

static void Pic2VecDiffAsciiFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	Pic2VecDiffAsciiImpl(args, result, false);
}
static void Pic2VecDiffAsciiThresholdFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	Pic2VecDiffAsciiImpl(args, result, true);
}

// ============================================
// SQL: pic_embed_batch(LIST(VARCHAR)) -> LIST(LIST(FLOAT))
// Embeds N paths in a single FFI call; saves overhead vs N separate
// pic_embed() invocations when processing thousands of images.
// ============================================
static void Pic2VecEmbedBatchFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	EnsureBundledLoaded();
	auto count = args.size();
	auto &input = args.data[0];
	input.Flatten(count);

	auto in_list_data = ListVector::GetData(input);
	auto &in_child = ListVector::GetEntry(input);
	auto in_child_data = FlatVector::GetData<string_t>(in_child);

	auto out_list_data = ListVector::GetData(result);
	idx_t row_offset = 0;
	idx_t inner_offset = 0;

	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(input, i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto entry = in_list_data[i];

		std::vector<std::string> path_storage;
		path_storage.reserve(entry.length);
		std::vector<const char *> ptrs;
		ptrs.reserve(entry.length);
		for (idx_t j = 0; j < entry.length; j++) {
			path_storage.emplace_back(in_child_data[entry.offset + j].GetString());
			ptrs.push_back(path_storage.back().c_str());
		}

		float *out_data = nullptr;
		size_t out_total_len = 0, per = 0;
		auto rc = pic2vec_rs_embed_batch(
		    ptrs.data(), ptrs.size(), nullptr,
		    &out_data, &out_total_len, &per);
		if (rc != 0) {
			throw IOException("pic_embed_batch: %s", LastRustError().c_str());
		}

		out_list_data[i].offset = row_offset;
		out_list_data[i].length = entry.length;

		ListVector::Reserve(result, row_offset + entry.length);
		auto &out_inner_list = ListVector::GetEntry(result);

		ListVector::Reserve(out_inner_list, inner_offset + out_total_len);
		auto inner_list_data = ListVector::GetData(out_inner_list);
		auto &inner_child = ListVector::GetEntry(out_inner_list);
		auto inner_child_data = FlatVector::GetData<float>(inner_child);

		for (size_t j = 0; j < ptrs.size(); j++) {
			inner_list_data[row_offset + j].offset = inner_offset + j * per;
			inner_list_data[row_offset + j].length = per;
		}
		std::memcpy(inner_child_data + inner_offset, out_data, out_total_len * sizeof(float));
		ListVector::SetListSize(out_inner_list, inner_offset + out_total_len);

		row_offset += entry.length;
		inner_offset += out_total_len;
		pic2vec_rs_free_floats(out_data, out_total_len);
	}
	ListVector::SetListSize(result, row_offset);
}

// ============================================
// SQL: pic_dedupe(path_a, path_b[, hamming_threshold]) -> BOOLEAN
// Quick perceptual-hash-based duplicate check. Faster than pic_match for
// near-exact duplicates (only resize+compare, no neural net).
// Default Hamming threshold: 5 (very strict; 0=identical, ~10=near-dup).
// ============================================
static constexpr int32_t DEFAULT_DEDUPE_HAMMING = 5;

static int32_t HammingU64(uint64_t a, uint64_t b) {
	return __builtin_popcountll(a ^ b);
}

static void Pic2VecDedupePathsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat af, bf, tf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);
	bool has_threshold = (args.ColumnCount() >= 3);
	if (has_threshold) args.data[2].ToUnifiedFormat(count, tf);

	auto rd = FlatVector::GetData<bool>(result);
	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto pa = ((string_t *)af.data)[ai].GetString();
		auto pb = ((string_t *)bf.data)[bi].GetString();

		uint64_t ha = 0, hb = 0;
		if (pic2vec_rs_phash(pa.c_str(), &ha) != 0 ||
		    pic2vec_rs_phash(pb.c_str(), &hb) != 0) {
			throw IOException("pic_dedupe: %s", LastRustError().c_str());
		}
		int32_t threshold = has_threshold
		    ? (int32_t)((int32_t *)tf.data)[tf.sel->get_index(i)]
		    : DEFAULT_DEDUPE_HAMMING;
		rd[i] = (HammingU64(ha, hb) <= threshold);
	}
}

// ============================================
// SQL: pic_match(vec_a, vec_b[, threshold]) / pic_match(path_a, path_b[, threshold])
// Returns BOOLEAN (cosine similarity >= threshold).
// ============================================

static constexpr double DEFAULT_MATCH_THRESHOLD = 0.85;

// Vector overload (with explicit threshold).
static void Pic2VecMatchVecFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<bool>(result);
	UnifiedVectorFormat tf;
	args.data[2].ToUnifiedFormat(count, tf);
	auto thresholds = (double *)tf.data;

	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(args.data[0], i) || FlatVector::IsNull(args.data[1], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto a = ExtractFloatList(args.data[0], i, count);
		auto b = ExtractFloatList(args.data[1], i, count);
		if (a.size() != b.size()) {
			throw InvalidInputException("pic_match: dims must match");
		}
		double threshold = thresholds[tf.sel->get_index(i)];
		rd[i] = (CosineSimilarity(a, b) >= threshold);
	}
}

// Vector overload (default threshold).
static void Pic2VecMatchVecDefaultFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<bool>(result);
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(args.data[0], i) || FlatVector::IsNull(args.data[1], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto a = ExtractFloatList(args.data[0], i, count);
		auto b = ExtractFloatList(args.data[1], i, count);
		if (a.size() != b.size()) {
			throw InvalidInputException("pic_match: dims must match");
		}
		rd[i] = (CosineSimilarity(a, b) >= DEFAULT_MATCH_THRESHOLD);
	}
}

// Helper: read embedding from a path via Rust FFI.
static std::vector<float> EmbedPath(const std::string &path) {
	EnsureBundledLoaded();
	float *out_data = nullptr;
	size_t out_len = 0;
	auto rc = pic2vec_rs_embed_path(path.c_str(), nullptr, &out_data, &out_len);
	if (rc != 0) {
		throw IOException("pic_match: %s", LastRustError().c_str());
	}
	std::vector<float> v(out_data, out_data + out_len);
	pic2vec_rs_free_floats(out_data, out_len);
	return v;
}

// Path overload (with explicit threshold).
static void Pic2VecMatchPathFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<bool>(result);
	UnifiedVectorFormat af, bf, tf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);
	args.data[2].ToUnifiedFormat(count, tf);
	auto thresholds = (double *)tf.data;

	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto pa = ((string_t *)af.data)[ai].GetString();
		auto pb = ((string_t *)bf.data)[bi].GetString();
		auto va = EmbedPath(pa);
		auto vb = EmbedPath(pb);
		double threshold = thresholds[tf.sel->get_index(i)];
		rd[i] = (CosineSimilarity(va, vb) >= threshold);
	}
}

// Path overload (default threshold).
static void Pic2VecMatchPathDefaultFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<bool>(result);
	UnifiedVectorFormat af, bf;
	args.data[0].ToUnifiedFormat(count, af);
	args.data[1].ToUnifiedFormat(count, bf);

	for (idx_t i = 0; i < count; i++) {
		auto ai = af.sel->get_index(i), bi = bf.sel->get_index(i);
		if (!af.validity.RowIsValid(ai) || !bf.validity.RowIsValid(bi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto pa = ((string_t *)af.data)[ai].GetString();
		auto pb = ((string_t *)bf.data)[bi].GetString();
		auto va = EmbedPath(pa);
		auto vb = EmbedPath(pb);
		rd[i] = (CosineSimilarity(va, vb) >= DEFAULT_MATCH_THRESHOLD);
	}
}

static void Pic2VecNormalizeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(args.data[0], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto vec = ExtractFloatList(args.data[0], i, count);
		double norm = 0;
		for (auto v : vec) norm += (double)v * v;
		norm = std::sqrt(norm);

		auto list_entry = ListVector::GetData(result);
		list_entry[i].offset = offset;
		list_entry[i].length = vec.size();

		idx_t new_size = offset + vec.size();
		ListVector::Reserve(result, new_size);
		auto &child = ListVector::GetEntry(result);
		auto child_data = FlatVector::GetData<float>(child);
		if (norm > 1e-12) {
			for (size_t j = 0; j < vec.size(); j++)
				child_data[offset + j] = static_cast<float>(vec[j] / norm);
		} else {
			for (size_t j = 0; j < vec.size(); j++) child_data[offset + j] = 0.0f;
		}
		offset = new_size;
	}
	ListVector::SetListSize(result, offset);
}

static void Pic2VecDimFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto rd = FlatVector::GetData<int32_t>(result);
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(args.data[0], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto list_data = ListVector::GetData(args.data[0]);
		rd[i] = static_cast<int32_t>(list_data[i].length);
	}
}

// ============================================
// SQL: pic2vec_download_model(name) -> VARCHAR
// (system curl + Rust load)
// ============================================

static const std::pair<const char *, const char *> KNOWN_MODELS[] = {
    {"vit_t16", "https://huggingface.co/eupe-onnx/eupe_vit_t16/resolve/main/eupe_vit_t16_fp16.onnx"},
    {"vit_s16", "https://huggingface.co/eupe-onnx/eupe_vit_s16/resolve/main/eupe_vit_s16_fp16.onnx"},
    {"vit_b16", "https://huggingface.co/eupe-onnx/eupe_vit_b16/resolve/main/eupe_vit_b16_fp16.onnx"},
};

static std::string LookupModelUrl(const std::string &name) {
	for (auto &e : KNOWN_MODELS) if (name == e.first) return e.second;
	return "";
}

static std::string HomeDir() {
	const char *h = std::getenv("HOME");
#ifdef _WIN32
	if (!h) h = std::getenv("USERPROFILE");
#endif
	return h ? h : ".";
}

static bool DirExists(const std::string &p) {
	struct stat st; return stat(p.c_str(), &st) == 0 && (st.st_mode & S_IFDIR);
}
static bool FileExists(const std::string &p) {
	struct stat st; return stat(p.c_str(), &st) == 0 && (st.st_mode & S_IFREG);
}
static void EnsureDir(const std::string &p) {
	if (DirExists(p)) return;
	auto slash = p.find_last_of("/\\");
	if (slash != std::string::npos && slash > 0) EnsureDir(p.substr(0, slash));
	MKDIR(p.c_str());
}

static std::string ResolveCachePath(const std::string &name) {
	const char *o = std::getenv("PIC2VEC_MODEL_DIR");
	std::string dir = (o && *o) ? std::string(o) : (HomeDir() + "/.duckdb/extensions/pic2vec/models");
	EnsureDir(dir);
	return dir + "/" + name + ".onnx";
}

static void Pic2VecDownloadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat nf;
	args.data[0].ToUnifiedFormat(count, nf);
	auto rd = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto ni = nf.sel->get_index(i);
		if (!nf.validity.RowIsValid(ni)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto name = ((string_t *)nf.data)[ni].GetString();
		auto url = LookupModelUrl(name);
		if (url.empty()) {
			rd[i] = StringVector::AddString(result, "Unknown model: " + name);
			continue;
		}
		auto out_path = ResolveCachePath(name);
		std::string msg;
		if (!FileExists(out_path)) {
			std::string cmd = "curl -fL --progress-bar -o '" + out_path + ".part' '" + url + "'";
			int rc = std::system(cmd.c_str());
			if (rc != 0) {
				std::remove((out_path + ".part").c_str());
				rd[i] = StringVector::AddString(result, "Download failed (curl exit " + std::to_string(rc) + ")");
				continue;
			}
			std::rename((out_path + ".part").c_str(), out_path.c_str());
			msg = "Downloaded to " + out_path + "; ";
		} else {
			msg = "Cached at " + out_path + "; ";
		}
		std::string load_msg;
		DoLoadModel(out_path, name, load_msg);
		rd[i] = StringVector::AddString(result, msg + load_msg);
	}
}

// ============================================
// Extension Load
// ============================================

void Pic2vecExtension::Load(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction(
	    "pic_version", {}, LogicalType::VARCHAR, Pic2VecVersionFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_bundled_info", {}, LogicalType::VARCHAR, Pic2VecBundledInfoFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_load_bundled", {}, LogicalType::VARCHAR, Pic2VecLoadBundledFunction));

	loader.RegisterFunction(ScalarFunction(
	    "pic_load_model", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    Pic2VecLoadModelSingleFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_load_model", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::VARCHAR, Pic2VecLoadModelFunction));

	loader.RegisterFunction(ScalarFunction(
	    "pic_unload_model", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    Pic2VecUnloadModelFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_list_models", {}, LogicalType::VARCHAR, Pic2VecListModelsFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_download_model", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    Pic2VecDownloadModelFunction));

	loader.RegisterFunction(ScalarFunction(
	    "pic_embed", {LogicalType::VARCHAR},
	    LogicalType::LIST(LogicalType::FLOAT), Pic2VecEmbedFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_embed", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::LIST(LogicalType::FLOAT), Pic2VecEmbedWithModelFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_embed_blob", {LogicalType::BLOB},
	    LogicalType::LIST(LogicalType::FLOAT), Pic2VecEmbedBlobFunction));

	loader.RegisterFunction(ScalarFunction(
	    "pic_similarity",
	    {LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::DOUBLE, Pic2VecSimilarityFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_distance",
	    {LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::DOUBLE, Pic2VecDistanceFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_inner_product",
	    {LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::DOUBLE, Pic2VecInnerProductFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_normalize", {LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::LIST(LogicalType::FLOAT), Pic2VecNormalizeFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_dim", {LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::INTEGER, Pic2VecDimFunction));

	// pic_image_info - returns STRUCT
	child_list_t<LogicalType> image_info_struct = {
	    {"width", LogicalType::INTEGER},
	    {"height", LogicalType::INTEGER},
	    {"channels", LogicalType::INTEGER},
	    {"format", LogicalType::VARCHAR},
	};
	loader.RegisterFunction(ScalarFunction(
	    "pic_image_info", {LogicalType::VARCHAR},
	    LogicalType::STRUCT(image_info_struct), Pic2VecImageInfoFunction));

	loader.RegisterFunction(ScalarFunction(
	    "pic_phash", {LogicalType::VARCHAR},
	    LogicalType::BIGINT, Pic2VecPhashFunction));

	loader.RegisterFunction(ScalarFunction(
	    "pic_hamming", {LogicalType::BIGINT, LogicalType::BIGINT},
	    LogicalType::INTEGER, Pic2VecHammingFunction));

	// pic_embed_batch
	loader.RegisterFunction(ScalarFunction(
	    "pic_embed_batch",
	    {LogicalType::LIST(LogicalType::VARCHAR)},
	    LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT)),
	    Pic2VecEmbedBatchFunction));

	// pic_diff_save - annotated copies with red rectangles
	loader.RegisterFunction(ScalarFunction(
	    "pic_diff_save",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::INTEGER, Pic2VecDiffSaveFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_diff_save",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE},
	    LogicalType::INTEGER, Pic2VecDiffSaveThresholdFunction));

	// pic_diff_heatmap - red gradient overlay (intensity ∝ 1-sim)
	loader.RegisterFunction(ScalarFunction(
	    "pic_diff_heatmap",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::BOOLEAN, Pic2VecDiffHeatmapFunction));

	// pic_diff_ascii - CLI-friendly grid visualization
	loader.RegisterFunction(ScalarFunction(
	    "pic_diff_ascii",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::VARCHAR, Pic2VecDiffAsciiFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_diff_ascii",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE},
	    LogicalType::VARCHAR, Pic2VecDiffAsciiThresholdFunction));

	// pic_diff_patches - returns LIST(STRUCT(row, col, sim))
	child_list_t<LogicalType> diff_struct = {
	    {"row", LogicalType::INTEGER},
	    {"col", LogicalType::INTEGER},
	    {"sim", LogicalType::DOUBLE},
	};
	loader.RegisterFunction(ScalarFunction(
	    "pic_diff_patches",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::LIST(LogicalType::STRUCT(diff_struct)),
	    Pic2VecDiffPatchesFunction));

	// pic_dedupe (path-based perceptual-hash duplicate check; 2 overloads)
	loader.RegisterFunction(ScalarFunction(
	    "pic_dedupe", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::BOOLEAN, Pic2VecDedupePathsFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_dedupe",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER},
	    LogicalType::BOOLEAN, Pic2VecDedupePathsFunction));

	// pic_match - 4 overloads (vec/path × default/threshold)
	loader.RegisterFunction(ScalarFunction(
	    "pic_match",
	    {LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::BOOLEAN, Pic2VecMatchVecDefaultFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_match",
	    {LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT), LogicalType::DOUBLE},
	    LogicalType::BOOLEAN, Pic2VecMatchVecFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_match",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::BOOLEAN, Pic2VecMatchPathDefaultFunction));
	loader.RegisterFunction(ScalarFunction(
	    "pic_match",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE},
	    LogicalType::BOOLEAN, Pic2VecMatchPathFunction));
}

std::string Pic2vecExtension::Name() { return "pic2vec"; }

std::string Pic2vecExtension::Version() const {
#ifdef EXT_VERSION_PIC2VEC
	return EXT_VERSION_PIC2VEC;
#else
	return "0.1.0";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void pic2vec_duckdb_cpp_init(duckdb::ExtensionLoader &loader) {
	duckdb::Pic2vecExtension ext;
	ext.Load(loader);
}

DUCKDB_EXTENSION_API const char *pic2vec_version() {
	return duckdb::DuckDB::LibraryVersion();
}

}
