#include "kvbound/mapped_file.h"

#include <utility>

#include "kvbound/error.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kvb {

#ifdef _WIN32

MappedFile::MappedFile(const std::filesystem::path& path) {
  const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) throw Error("cannot open " + path.string());

  LARGE_INTEGER size{};
  const bool sized = GetFileSizeEx(file, &size) != 0 && size.QuadPart > 0;
  // The view keeps the file alive, so both handles close as soon as the view exists.
  const HANDLE mapping =
      sized ? CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
  CloseHandle(file);
  if (mapping == nullptr) throw Error("cannot map " + path.string() + " (empty or unreadable)");

  const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(mapping);
  if (view == nullptr) throw Error("cannot map " + path.string());

  data_ = static_cast<const std::byte*>(view);
  size_ = static_cast<std::size_t>(size.QuadPart);
}

void MappedFile::unmap() noexcept {
  if (data_ != nullptr) UnmapViewOfFile(data_);
}

#else

MappedFile::MappedFile(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw Error("cannot open " + path.string());

  struct stat st{};
  const bool sized = ::fstat(fd, &st) == 0 && st.st_size > 0;
  // The mapping keeps the file alive, so the descriptor closes as soon as the mapping exists.
  void* view =
      sized ? ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0)
            : MAP_FAILED;
  ::close(fd);
  if (view == MAP_FAILED) throw Error("cannot map " + path.string() + " (empty or unreadable)");

  data_ = static_cast<const std::byte*>(view);
  size_ = static_cast<std::size_t>(st.st_size);
}

void MappedFile::unmap() noexcept {
  if (data_ != nullptr) ::munmap(const_cast<std::byte*>(data_), size_);
}

#endif

MappedFile::~MappedFile() { unmap(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    unmap();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

}  // namespace kvb
