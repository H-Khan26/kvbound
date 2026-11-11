// JSON parser and safetensors loader: accepted syntax and rejection of malformed files.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "kvbound/dtype.h"
#include "kvbound/error.h"
#include "kvbound/json.h"
#include "kvbound/safetensors.h"

namespace {

using kvb::json::parse;

TEST(Json, ParsesNestedDocument) {
  const auto doc =
      parse(R"({"a": [1, -2.5, 1e3, true, null], "b": {"c": "x\n\u00e9\ud83d\ude00"}})");
  const auto& a = doc.at("a").as_array();
  EXPECT_EQ(a[0].as_int(), 1);
  EXPECT_DOUBLE_EQ(a[1].as_number(), -2.5);
  EXPECT_DOUBLE_EQ(a[2].as_number(), 1000.0);
  EXPECT_TRUE(a[3].as_bool());
  EXPECT_TRUE(a[4].is_null());
  EXPECT_EQ(doc.at("b").at("c").as_string(), "x\n\xC3\xA9\xF0\x9F\x98\x80");
  EXPECT_EQ(doc.find("missing"), nullptr);
}

TEST(Json, KeepsLargeOffsetsExact) {
  // Tensor byte offsets exceed 2^53 only in theory, but must never round through a double.
  EXPECT_EQ(parse("9007199254740993").as_int(), 9007199254740993LL);
}

TEST(Json, RejectsMalformedInput) {
  for (const char* bad :
       {"", "{", "[1,]", "{\"a\":1,}", "{\"a\":1,\"a\":2}", "01", "1.", "\"\\x\"", "\"\\ud800\"",
        "nul", "1e400", "1 2", "\"unterminated", "{\"a\" 1}", "-"}) {
    EXPECT_THROW(parse(bad), kvb::Error) << bad;
  }
  EXPECT_THROW(parse(std::string(100, '[')), kvb::Error);
  EXPECT_THROW(parse("1").as_string(), kvb::Error);
}

TEST(Dtype, HalfToFloat) {
  EXPECT_EQ(kvb::half_bits_to_float(0x3C00), 1.0f);
  EXPECT_EQ(kvb::half_bits_to_float(0xC000), -2.0f);
  EXPECT_EQ(kvb::half_bits_to_float(0x7BFF), 65504.0f);
  EXPECT_EQ(kvb::half_bits_to_float(0x0001), 0x1p-24f);  // smallest subnormal
  EXPECT_EQ(kvb::half_bits_to_float(0x0400), 0x1p-14f);  // smallest normal
}

// Writes a file made of an 8-byte header length, the header text and the data bytes, under a name
// unique to this test; removes it when destroyed.
class TempSafetensors {
 public:
  TempSafetensors(std::uint64_t header_len, const std::string& header, const std::string& data) {
    const auto* test = testing::UnitTest::GetInstance()->current_test_info();
    path_ = std::filesystem::temp_directory_path() /
            (std::string("kvbound_") + test->test_suite_name() + "_" + test->name() + "_" +
             std::to_string(counter_++) + ".safetensors");
    std::ofstream out(path_, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out << header << data;
    out.close();
    EXPECT_TRUE(out.good()) << "cannot write " << path_;
  }
  TempSafetensors(const std::string& header, const std::string& data)
      : TempSafetensors(header.size(), header, data) {}
  ~TempSafetensors() { std::filesystem::remove(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  static inline int counter_ = 0;
  std::filesystem::path path_;
};

TEST(Safetensors, ReadsTensorsAndMetadata) {
  const float values[2] = {1.5f, -2.0f};
  const std::string data(reinterpret_cast<const char*>(values), sizeof(values));
  const TempSafetensors tmp(
      R"({"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,1],"data_offsets":[0,8]}})",
      data);
  const kvb::SafetensorsFile file(tmp.path());
  EXPECT_EQ(file.metadata().at("format"), "pt");
  const kvb::TensorView& w = file.at("w");
  EXPECT_EQ(w.dtype, kvb::DType::F32);
  EXPECT_EQ(w.shape, (std::vector<std::int64_t>{2, 1}));
  float read[2];
  std::memcpy(read, w.data, sizeof(read));
  EXPECT_EQ(read[0], 1.5f);
  EXPECT_EQ(read[1], -2.0f);
  EXPECT_THROW(file.at("missing"), kvb::Error);
}

TEST(Safetensors, RejectsInconsistentHeaders) {
  const std::string data(8, '\0');
  for (const char* header : {
           R"({"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})",  // past end of file
           R"({"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})",   // size mismatch
           R"({"w":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})",  // negative dimension
           R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})",   // reversed offsets
           R"({"w":{"dtype":"F99","shape":[2],"data_offsets":[0,8]}})",   // unknown dtype
           R"({"w":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,8]}})",
           R"({"w":{"dtype":"F32","shape":[2]}})",  // missing offsets
           R"({"w":)",                              // truncated JSON
       }) {
    const TempSafetensors tmp(header, data);
    EXPECT_THROW(kvb::SafetensorsFile{tmp.path()}, kvb::Error) << header;
  }
}

TEST(Safetensors, RejectsHeaderLengthBeyondFile) {
  const TempSafetensors tmp(1000, "{}", "");
  EXPECT_THROW(kvb::SafetensorsFile{tmp.path()}, kvb::Error);
}

}  // namespace
