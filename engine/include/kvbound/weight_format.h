#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kvb {

// How a checkpoint stores its weight matrices. kvbound.quantize writes the format into the
// safetensors "__metadata__" map and python/kvbound/weight_format.py mirrors this header.
//
// A quantized file keeps every tensor name. Each matrix W [out, in] that activations are
// multiplied by becomes integers q plus a companion tensor "<name>.scale" of FP16 scales; every
// other tensor stays FP16 (biases, norm weights, convolution weights, position tables).
enum class WeightScheme : std::uint8_t {
  F16,  // floating point as stored (FP16, BF16 or FP32, held as FP16 on the device)
  // I8 [out, in] with scale [out]: W[o][i] = scale[o] * q[o][i], q in [-127, 127].
  Int8PerChannel,
  // U8 [out, in / 2] with scale [out, in / group_size]: W[o][i] = scale[o][i / group_size] *
  // q[o][i]. Byte j of a row holds q[2j] in its low four bits and q[2j + 1] in its high four,
  // each a two's-complement integer in [-7, 7].
  Int4Grouped,
};

// Names as written in the metadata: "fp16", "int8_pc", "int4_g".
std::string_view weight_scheme_name(WeightScheme scheme) noexcept;
std::optional<WeightScheme> weight_scheme_from_name(std::string_view name) noexcept;

struct WeightFormat {
  WeightScheme scheme = WeightScheme::F16;
  int group_size = 0;  // Int4Grouped: consecutive inputs that share one scale
};

// Reads the "kvbound.*" metadata keys. A file without "kvbound.scheme" holds floating-point
// tensors only. Throws kvb::Error on a version, scheme, packing or scale type this build does
// not read.
WeightFormat weight_format_from_metadata(
    const std::map<std::string, std::string, std::less<>>& metadata);

// Throws kvb::Error when the format cannot hold rows of `in` weights: Int8PerChannel needs
// in % 16 == 0, Int4Grouped needs a group size that is a power of two, at least 32, and divides
// in. The kernels read 16 bytes at a time and find a load's scale with a shift.
void check_row_length(const WeightFormat& format, std::int64_t in);

// Shapes a matrix [out, in] and its scales have in a file of this format, after
// check_row_length().
std::vector<std::int64_t> stored_shape(const WeightFormat& format, std::int64_t out,
                                       std::int64_t in);
std::vector<std::int64_t> scale_shape(const WeightFormat& format, std::int64_t out,
                                      std::int64_t in);

// One matrix on the device, as the kernels receive it.
struct WeightRef {
  WeightScheme scheme = WeightScheme::F16;
  const void* data = nullptr;            // FP16 elements, INT8 values, or packed 4-bit pairs
  const std::uint16_t* scale = nullptr;  // FP16 scales; nullptr for FP16 weights
  int group_size = 0;
};

}  // namespace kvb
