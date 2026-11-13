// The weight format read from safetensors metadata, and the shapes it gives stored matrices.

#include "kvbound/weight_format.h"

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "kvbound/error.h"

namespace {

using kvb::WeightFormat;
using kvb::WeightScheme;
using Metadata = std::map<std::string, std::string, std::less<>>;

Metadata int4_metadata(const std::string& group_size) {
  return {{"kvbound.format_version", "1"},    {"kvbound.scheme", "int4_g"},
          {"kvbound.group_size", group_size}, {"kvbound.pack_order", "lo_nibble_first"},
          {"kvbound.scale_dtype", "F16"},     {"kvbound.clip", "minmax"}};
}

TEST(WeightFormat, FileWithoutASchemeHoldsFloats) {
  const WeightFormat format = kvb::weight_format_from_metadata({{"format", "pt"}});
  EXPECT_EQ(format.scheme, WeightScheme::F16);
  EXPECT_EQ(kvb::stored_shape(format, 8, 100), (std::vector<std::int64_t>{8, 100}));
  EXPECT_TRUE(kvb::scale_shape(format, 8, 100).empty());
}

TEST(WeightFormat, ReadsBothQuantizedSchemes) {
  const WeightFormat int8 = kvb::weight_format_from_metadata({{"kvbound.format_version", "1"},
                                                              {"kvbound.scheme", "int8_pc"},
                                                              {"kvbound.group_size", "0"},
                                                              {"kvbound.pack_order", ""},
                                                              {"kvbound.scale_dtype", "F16"}});
  EXPECT_EQ(int8.scheme, WeightScheme::Int8PerChannel);
  EXPECT_EQ(kvb::stored_shape(int8, 5632, 2048), (std::vector<std::int64_t>{5632, 2048}));
  EXPECT_EQ(kvb::scale_shape(int8, 5632, 2048), (std::vector<std::int64_t>{5632}));

  const WeightFormat int4 = kvb::weight_format_from_metadata(int4_metadata("128"));
  EXPECT_EQ(int4.scheme, WeightScheme::Int4Grouped);
  EXPECT_EQ(int4.group_size, 128);
  // Two weights per byte, one scale per 128 weights.
  EXPECT_EQ(kvb::stored_shape(int4, 2048, 5632), (std::vector<std::int64_t>{2048, 2816}));
  EXPECT_EQ(kvb::scale_shape(int4, 2048, 5632), (std::vector<std::int64_t>{2048, 44}));
}

TEST(WeightFormat, NamesRoundTrip) {
  for (const WeightScheme scheme :
       {WeightScheme::F16, WeightScheme::Int8PerChannel, WeightScheme::Int4Grouped}) {
    EXPECT_EQ(kvb::weight_scheme_from_name(kvb::weight_scheme_name(scheme)), scheme);
  }
  EXPECT_FALSE(kvb::weight_scheme_from_name("int2"));
}

TEST(WeightFormat, RejectsWhatThisBuildCannotRead) {
  Metadata metadata = int4_metadata("128");
  metadata["kvbound.format_version"] = "2";
  EXPECT_THROW(kvb::weight_format_from_metadata(metadata), kvb::Error);
  metadata = int4_metadata("128");
  metadata["kvbound.scheme"] = "int2";
  EXPECT_THROW(kvb::weight_format_from_metadata(metadata), kvb::Error);
  metadata = int4_metadata("128");
  metadata["kvbound.pack_order"] = "hi_nibble_first";
  EXPECT_THROW(kvb::weight_format_from_metadata(metadata), kvb::Error);
  metadata = int4_metadata("128");
  metadata["kvbound.scale_dtype"] = "F32";
  EXPECT_THROW(kvb::weight_format_from_metadata(metadata), kvb::Error);
  EXPECT_THROW(kvb::weight_format_from_metadata(int4_metadata("12x")), kvb::Error);
  EXPECT_THROW(kvb::weight_format_from_metadata(int4_metadata("0")), kvb::Error);
}

TEST(WeightFormat, RejectsRowsTheKernelsCannotRead) {
  // The kernels read 16 bytes at a time and find an INT4 load's scale with a shift.
  const WeightFormat int8{.scheme = WeightScheme::Int8PerChannel};
  EXPECT_THROW(kvb::stored_shape(int8, 4, 100), kvb::Error);
  EXPECT_THROW(kvb::stored_shape({.scheme = WeightScheme::Int4Grouped, .group_size = 16}, 4, 64),
               kvb::Error);
  EXPECT_THROW(kvb::stored_shape({.scheme = WeightScheme::Int4Grouped, .group_size = 96}, 4, 192),
               kvb::Error);
  EXPECT_THROW(kvb::stored_shape({.scheme = WeightScheme::Int4Grouped, .group_size = 64}, 4, 96),
               kvb::Error);
  EXPECT_NO_THROW(
      kvb::stored_shape({.scheme = WeightScheme::Int4Grouped, .group_size = 32}, 4, 64));
}

}  // namespace
