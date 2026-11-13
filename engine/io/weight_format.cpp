#include "kvbound/weight_format.h"

#include <bit>
#include <charconv>

#include "kvbound/error.h"

namespace kvb {
namespace {

constexpr std::string_view kVersionKey = "kvbound.format_version";
constexpr std::string_view kSchemeKey = "kvbound.scheme";
constexpr std::string_view kGroupSizeKey = "kvbound.group_size";
constexpr std::string_view kPackOrderKey = "kvbound.pack_order";
constexpr std::string_view kScaleDtypeKey = "kvbound.scale_dtype";

// Weights one 16-byte load covers; a group of Int4Grouped must hold whole loads.
constexpr int kInt8PerLoad = 16;
constexpr int kInt4PerLoad = 32;

}  // namespace

std::string_view weight_scheme_name(WeightScheme scheme) noexcept {
  switch (scheme) {
    case WeightScheme::F16:
      return "fp16";
    case WeightScheme::Int8PerChannel:
      return "int8_pc";
    case WeightScheme::Int4Grouped:
      return "int4_g";
  }
  return "";
}

std::optional<WeightScheme> weight_scheme_from_name(std::string_view name) noexcept {
  for (const WeightScheme scheme :
       {WeightScheme::F16, WeightScheme::Int8PerChannel, WeightScheme::Int4Grouped}) {
    if (name == weight_scheme_name(scheme)) return scheme;
  }
  return std::nullopt;
}

WeightFormat weight_format_from_metadata(
    const std::map<std::string, std::string, std::less<>>& metadata) {
  const auto value = [&](std::string_view key) -> std::string_view {
    const auto it = metadata.find(key);
    return it == metadata.end() ? std::string_view{} : std::string_view{it->second};
  };
  WeightFormat format;
  if (!metadata.contains(kSchemeKey)) return format;

  if (value(kVersionKey) != "1") {
    throw Error("weight format version \"" + std::string(value(kVersionKey)) +
                "\" is not supported");
  }
  const auto scheme = weight_scheme_from_name(value(kSchemeKey));
  if (!scheme) throw Error("unknown weight scheme \"" + std::string(value(kSchemeKey)) + "\"");
  format.scheme = *scheme;
  if (format.scheme == WeightScheme::F16) return format;

  if (value(kScaleDtypeKey) != "F16") throw Error("weight scales must be F16");
  if (format.scheme == WeightScheme::Int4Grouped) {
    if (value(kPackOrderKey) != "lo_nibble_first") {
      throw Error("unknown 4-bit pack order \"" + std::string(value(kPackOrderKey)) + "\"");
    }
    const std::string_view text = value(kGroupSizeKey);
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), format.group_size);
    if (error != std::errc{} || end != text.data() + text.size() || format.group_size <= 0) {
      throw Error("bad weight group size \"" + std::string(text) + "\"");
    }
  }
  return format;
}

void check_row_length(const WeightFormat& format, std::int64_t in) {
  switch (format.scheme) {
    case WeightScheme::F16:
      break;
    case WeightScheme::Int8PerChannel:
      if (in % kInt8PerLoad != 0) throw Error("int8_pc needs a row length divisible by 16");
      break;
    case WeightScheme::Int4Grouped:
      if (format.group_size < kInt4PerLoad ||
          !std::has_single_bit(static_cast<unsigned>(format.group_size)) ||
          in % format.group_size != 0) {
        throw Error(
            "int4_g needs a group size that is a power of two, at least 32, and divides "
            "the row");
      }
      break;
  }
}

std::vector<std::int64_t> stored_shape(const WeightFormat& format, std::int64_t out,
                                       std::int64_t in) {
  check_row_length(format, in);
  return {out, format.scheme == WeightScheme::Int4Grouped ? in / 2 : in};
}

std::vector<std::int64_t> scale_shape(const WeightFormat& format, std::int64_t out,
                                      std::int64_t in) {
  switch (format.scheme) {
    case WeightScheme::F16:
      return {};
    case WeightScheme::Int8PerChannel:
      return {out};
    case WeightScheme::Int4Grouped:
      return {out, in / format.group_size};
  }
  return {};
}

}  // namespace kvb
