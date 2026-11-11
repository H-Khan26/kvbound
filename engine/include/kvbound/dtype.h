#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace kvb {

// Element types stored in weights, activations and the KV cache.
// Sub-byte formats (INT4) are packed into U8 and described by the weight format, not here.
enum class DType : std::uint8_t { F32, F16, BF16, F8E4M3, I8, U8, I32, I64 };

constexpr std::size_t size_bytes(DType t) noexcept {
  switch (t) {
    case DType::F32:
    case DType::I32:
      return 4;
    case DType::F16:
    case DType::BF16:
      return 2;
    case DType::F8E4M3:
    case DType::I8:
    case DType::U8:
      return 1;
    case DType::I64:
      return 8;
  }
  return 0;
}

// Converts IEEE 754 binary16 bits to float exactly (every half is representable as a float).
// Host-side only: tests and loaders use it; kernels convert on the GPU.
float half_bits_to_float(std::uint16_t bits) noexcept;

// Names as written in safetensors headers: "F32", "F16", "BF16", "F8_E4M3", "I8", ...
std::string_view dtype_name(DType t) noexcept;
std::optional<DType> dtype_from_name(std::string_view name) noexcept;

}  // namespace kvb
