#include "pic2vec_extension.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

// ============================================
// stb_image - single-header image loading
// Bundled inline to avoid external dependency
// ============================================
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_NO_STDIO
#endif

// Minimal stb_image declarations for our use case
// In production, include the full stb_image.h header.
// For now we implement a simple image decoder.

namespace {

// ImageNet normalization constants
constexpr float IMAGENET_MEAN[] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[] = {0.229f, 0.224f, 0.225f};

// Simple bilinear resize
void BilinearResize(const uint8_t *src, int src_w, int src_h, int channels,
                    uint8_t *dst, int dst_w, int dst_h) {
	for (int y = 0; y < dst_h; y++) {
		float fy = (float)y * src_h / dst_h;
		int y0 = (int)fy;
		int y1 = std::min(y0 + 1, src_h - 1);
		float wy = fy - y0;

		for (int x = 0; x < dst_w; x++) {
			float fx = (float)x * src_w / dst_w;
			int x0 = (int)fx;
			int x1 = std::min(x0 + 1, src_w - 1);
			float wx = fx - x0;

			for (int c = 0; c < channels; c++) {
				float v00 = src[(y0 * src_w + x0) * channels + c];
				float v01 = src[(y0 * src_w + x1) * channels + c];
				float v10 = src[(y1 * src_w + x0) * channels + c];
				float v11 = src[(y1 * src_w + x1) * channels + c];

				float v = v00 * (1 - wx) * (1 - wy) +
				          v01 * wx * (1 - wy) +
				          v10 * (1 - wx) * wy +
				          v11 * wx * wy;

				dst[(y * dst_w + x) * channels + c] = (uint8_t)std::min(255.0f, std::max(0.0f, v));
			}
		}
	}
}

// Convert HWC uint8 image to CHW float with ImageNet normalization
std::vector<float> HWCtoNCHW(const uint8_t *data, int width, int height, int channels) {
	// Output: [1, 3, H, W] = [3, H, W] for single batch
	std::vector<float> output(3 * height * width);

	for (int c = 0; c < 3; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int src_c = (channels >= 3) ? c : 0; // grayscale -> replicate
				float pixel = data[(y * width + x) * channels + src_c] / 255.0f;
				// ImageNet normalization
				pixel = (pixel - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
				output[c * height * width + y * width + x] = pixel;
			}
		}
	}
	return output;
}

// ============================================
// Minimal PNG decoder (supports 8-bit RGB/RGBA)
// For production, use stb_image.h or libpng
// ============================================

struct RawImage {
	std::vector<uint8_t> pixels; // HWC format
	int width = 0;
	int height = 0;
	int channels = 0;
};

// Detect image format from magic bytes
enum class ImageFormat { UNKNOWN, PNG, JPEG, BMP };

ImageFormat DetectFormat(const uint8_t *data, size_t size) {
	if (size < 4) return ImageFormat::UNKNOWN;
	// PNG: 89 50 4E 47
	if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) {
		return ImageFormat::PNG;
	}
	// JPEG: FF D8 FF
	if (data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
		return ImageFormat::JPEG;
	}
	// BMP: 42 4D
	if (data[0] == 0x42 && data[1] == 0x4D) {
		return ImageFormat::BMP;
	}
	return ImageFormat::UNKNOWN;
}

// We rely on stb_image for actual decoding.
// This file uses stb_image as a compile-time dependency.
// The stb_image.h header should be placed in src/include/

} // anonymous namespace

// ============================================
// stb_image integration
// We use a simplified approach: read file -> decode with stb
// If stb_image.h is not available, provide a stub that
// throws an error directing user to install it.
// ============================================

// Try to include stb_image.h - if not found, use our own decoder
#if __has_include("stb_image.h")
#include "stb_image.h"
#define HAS_STB_IMAGE 1
#else
#define HAS_STB_IMAGE 0
#endif

namespace duckdb {

static RawImage DecodeImageFromMemory(const uint8_t *data, size_t size) {
	RawImage img;

#if HAS_STB_IMAGE
	int w, h, c;
	// Force 3 channels (RGB)
	uint8_t *pixels = stbi_load_from_memory(data, (int)size, &w, &h, &c, 3);
	if (!pixels) {
		throw std::runtime_error("Failed to decode image: " + std::string(stbi_failure_reason()));
	}
	img.width = w;
	img.height = h;
	img.channels = 3;
	img.pixels.assign(pixels, pixels + w * h * 3);
	stbi_image_free(pixels);
#else
	// Fallback: minimal BMP decoder for testing, or throw
	auto format = DetectFormat(data, size);

	if (format == ImageFormat::BMP && size > 54) {
		// Minimal BMP decoder (uncompressed 24-bit)
		int offset = *(int *)(data + 10);
		int w = *(int *)(data + 18);
		int h = *(int *)(data + 22);
		int bpp = *(short *)(data + 28);

		if (bpp == 24 && w > 0 && h > 0) {
			img.width = w;
			img.height = std::abs(h);
			img.channels = 3;
			img.pixels.resize(img.width * img.height * 3);

			int row_size = ((w * 3 + 3) / 4) * 4; // BMP rows are 4-byte aligned
			bool bottom_up = (h > 0);

			for (int y = 0; y < img.height; y++) {
				int src_y = bottom_up ? (img.height - 1 - y) : y;
				const uint8_t *row = data + offset + src_y * row_size;
				for (int x = 0; x < img.width; x++) {
					// BMP is BGR
					img.pixels[(y * img.width + x) * 3 + 0] = row[x * 3 + 2]; // R
					img.pixels[(y * img.width + x) * 3 + 1] = row[x * 3 + 1]; // G
					img.pixels[(y * img.width + x) * 3 + 2] = row[x * 3 + 0]; // B
				}
			}
		} else {
			throw std::runtime_error("Unsupported BMP format (only 24-bit uncompressed supported without stb_image)");
		}
	} else {
		throw std::runtime_error(
		    "Image decoding requires stb_image.h. "
		    "Place stb_image.h in src/include/ or install via vcpkg. "
		    "Without it, only 24-bit BMP is supported.");
	}
#endif

	return img;
}

ImageData LoadAndPreprocessImage(const std::string &file_path, int target_h, int target_w) {
	// Read file
	std::ifstream file(file_path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open image file: " + file_path);
	}

	auto size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> buffer(size);
	if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
		throw std::runtime_error("Failed to read image file: " + file_path);
	}

	return LoadAndPreprocessImageFromMemory(buffer.data(), buffer.size(), target_h, target_w);
}

ImageData LoadAndPreprocessImageFromMemory(const uint8_t *data, size_t size, int target_h, int target_w) {
	// Decode
	auto raw = DecodeImageFromMemory(data, size);

	// Resize if needed
	std::vector<uint8_t> resized;
	const uint8_t *src_pixels = raw.pixels.data();
	int src_w = raw.width;
	int src_h = raw.height;

	if (src_w != target_w || src_h != target_h) {
		resized.resize(target_w * target_h * raw.channels);
		BilinearResize(raw.pixels.data(), raw.width, raw.height, raw.channels,
		               resized.data(), target_w, target_h);
		src_pixels = resized.data();
		src_w = target_w;
		src_h = target_h;
	}

	// Convert to CHW float with ImageNet normalization
	ImageData result;
	result.width = target_w;
	result.height = target_h;
	result.channels = 3;
	result.pixels = HWCtoNCHW(src_pixels, target_w, target_h, raw.channels);

	return result;
}

} // namespace duckdb
