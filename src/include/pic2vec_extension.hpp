#pragma once

#include "duckdb/main/extension.hpp"

#include <string>
#include <vector>

namespace duckdb {

// ============================================
// Vector operations (pure C++ - no FFI overhead for trivial math)
// ============================================

double CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);
double L2Distance(const std::vector<float> &a, const std::vector<float> &b);
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
