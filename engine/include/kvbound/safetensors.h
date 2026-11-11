#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>

#include "kvbound/mapped_file.h"
#include "kvbound/tensor.h"

namespace kvb {

// A memory-mapped safetensors file. Tensor views point into the mapping and stay valid for the
// lifetime of this object.
//
// Layout: 8-byte little-endian header length N, N bytes of JSON, then raw tensor data. Each
// header entry gives a dtype, a shape and [begin, end) byte offsets into the data section; the
// optional "__metadata__" entry is a string-to-string map.
class SafetensorsFile {
 public:
  using TensorMap = std::map<std::string, TensorView, std::less<>>;
  using Metadata = std::map<std::string, std::string, std::less<>>;

  // Throws kvb::Error if the file is missing, truncated, or its header is inconsistent.
  explicit SafetensorsFile(const std::filesystem::path& path);

  const TensorMap& tensors() const noexcept { return tensors_; }
  const Metadata& metadata() const noexcept { return metadata_; }

  // Returns nullptr when no tensor has this name.
  const TensorView* find(std::string_view name) const;
  // Throws kvb::Error when no tensor has this name.
  const TensorView& at(std::string_view name) const;

 private:
  MappedFile file_;
  TensorMap tensors_;
  Metadata metadata_;
};

}  // namespace kvb
