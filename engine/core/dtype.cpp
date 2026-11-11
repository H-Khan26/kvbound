#include "kvbound/dtype.h"

#include <array>
#include <bit>
#include <utility>

namespace kvb {
namespace {

constexpr std::array<std::pair<DType, std::string_view>, 8> kNames{{
    {DType::F32, "F32"},
    {DType::F16, "F16"},
    {DType::BF16, "BF16"},
    {DType::F8E4M3, "F8_E4M3"},
    {DType::I8, "I8"},
    {DType::U8, "U8"},
    {DType::I32, "I32"},
    {DType::I64, "I64"},
}};

}  // namespace

float half_bits_to_float(std::uint16_t bits) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  std::uint32_t exponent = (bits >> 10) & 0x1Fu;
  std::uint32_t mantissa = bits & 0x3FFu;

  std::uint32_t out;
  if (exponent == 0x1Fu) {  // infinity or NaN
    out = sign | 0x7F800000u | (mantissa << 13);
  } else if (exponent != 0) {  // normal: rebias the exponent from 15 to 127
    out = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  } else if (mantissa == 0) {  // signed zero
    out = sign;
  } else {  // subnormal: shift the leading one up to the implicit position
    exponent = 113u;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    out = sign | (exponent << 23) | ((mantissa & 0x3FFu) << 13);
  }
  return std::bit_cast<float>(out);
}

std::string_view dtype_name(DType t) noexcept {
  for (const auto& [dtype, name] : kNames) {
    if (dtype == t) return name;
  }
  return "?";
}

std::optional<DType> dtype_from_name(std::string_view name) noexcept {
  for (const auto& [dtype, n] : kNames) {
    if (n == name) return dtype;
  }
  return std::nullopt;
}

}  // namespace kvb
