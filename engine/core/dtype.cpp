#include "kvbound/dtype.h"

#include <array>
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
