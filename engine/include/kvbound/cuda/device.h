#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "kvbound/error.h"

namespace kvb::cuda {

// Throws kvb::Error naming the failed call when status is not cudaSuccess.
void check(cudaError_t status, const char* what);

#define KVB_CUDA(call) ::kvb::cuda::check((call), #call)

// Owning device allocation. Move-only.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
    return *this;
  }

  std::size_t bytes() const noexcept { return bytes_; }
  template <typename T = void>
  T* as() const noexcept {
    return static_cast<T*>(data_);
  }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

// Owning page-locked host allocation, so copies to and from the device are asynchronous.
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t bytes);
  ~PinnedBuffer();
  PinnedBuffer(PinnedBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
    return *this;
  }

  std::size_t bytes() const noexcept { return bytes_; }
  template <typename T = void>
  T* as() const noexcept {
    return static_cast<T*>(data_);
  }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace kvb::cuda
