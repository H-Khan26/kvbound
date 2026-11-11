#include "kvbound/safetensors.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>

#include "kvbound/error.h"
#include "kvbound/json.h"

namespace kvb {
namespace {

static_assert(std::endian::native == std::endian::little, "safetensors data is little-endian");

constexpr std::size_t kLengthPrefix = 8;
constexpr std::uint64_t kMaxHeaderBytes = 100u << 20;  // same cap as the reference implementation

// Product of the dimensions times the element size, or throws on negative sizes and overflow.
std::uint64_t byte_size(const json::Array& shape, DType dtype, std::vector<std::int64_t>& dims) {
  std::uint64_t bytes = size_bytes(dtype);
  for (const json::Value& dim : shape) {
    const std::int64_t n = dim.as_int();
    if (n < 0) throw Error("negative dimension");
    const auto un = static_cast<std::uint64_t>(n);
    if (un != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / un) {
      throw Error("shape overflows");
    }
    bytes *= un;
    dims.push_back(n);
  }
  return bytes;
}

TensorView parse_entry(const json::Value& entry, std::span<const std::byte> data) {
  const std::string& dtype_text = entry.at("dtype").as_string();
  const auto dtype = dtype_from_name(dtype_text);
  if (!dtype) throw Error("unsupported dtype " + dtype_text);

  TensorView view;
  view.dtype = *dtype;
  const std::uint64_t expected = byte_size(entry.at("shape").as_array(), *dtype, view.shape);

  const json::Array& offsets = entry.at("data_offsets").as_array();
  if (offsets.size() != 2) throw Error("data_offsets must have two entries");
  const std::int64_t begin = offsets[0].as_int();
  const std::int64_t end = offsets[1].as_int();
  if (begin < 0 || end < begin || static_cast<std::uint64_t>(end) > data.size()) {
    throw Error("data_offsets fall outside the file");
  }
  if (static_cast<std::uint64_t>(end - begin) != expected) {
    throw Error("data_offsets do not match shape and dtype");
  }
  view.data = data.data() + begin;
  view.nbytes = static_cast<std::size_t>(expected);
  return view;
}

}  // namespace

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : file_(path) {
  const std::span<const std::byte> bytes = file_.bytes();
  const std::string name = path.string();
  if (bytes.size() < kLengthPrefix) throw Error(name + ": too short to be a safetensors file");

  std::uint64_t header_len = 0;
  std::memcpy(&header_len, bytes.data(), kLengthPrefix);
  if (header_len > kMaxHeaderBytes || header_len > bytes.size() - kLengthPrefix) {
    throw Error(name + ": header length exceeds the file");
  }
  const auto header_bytes = static_cast<std::size_t>(header_len);
  const std::string_view header_text = file_.text().substr(kLengthPrefix, header_bytes);
  const std::span<const std::byte> data = bytes.subspan(kLengthPrefix + header_bytes);

  try {
    const json::Value header = json::parse(header_text);
    for (const auto& [key, entry] : header.as_object()) {
      if (key == "__metadata__") {
        for (const auto& [meta_key, meta_value] : entry.as_object()) {
          metadata_.emplace(meta_key, meta_value.as_string());
        }
        continue;
      }
      try {
        tensors_.emplace(key, parse_entry(entry, data));
      } catch (const Error& e) {
        throw Error("tensor \"" + key + "\": " + e.what());
      }
    }
  } catch (const Error& e) {
    throw Error(name + ": " + e.what());
  }
}

const TensorView* SafetensorsFile::find(std::string_view name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& SafetensorsFile::at(std::string_view name) const {
  if (const TensorView* view = find(name)) return *view;
  throw Error("no tensor named \"" + std::string(name) + "\"");
}

}  // namespace kvb
