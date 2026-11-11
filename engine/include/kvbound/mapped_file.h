#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>

namespace kvb {

// Read-only memory mapping of a whole file. Move-only; unmaps on destruction.
class MappedFile {
 public:
  // Throws kvb::Error if the file cannot be opened or is empty.
  explicit MappedFile(const std::filesystem::path& path);
  ~MappedFile();

  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  std::span<const std::byte> bytes() const noexcept { return {data_, size_}; }
  std::string_view text() const noexcept { return {reinterpret_cast<const char*>(data_), size_}; }

 private:
  void unmap() noexcept;

  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace kvb
