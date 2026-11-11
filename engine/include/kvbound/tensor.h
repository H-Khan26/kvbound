#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "kvbound/dtype.h"
#include "kvbound/error.h"

namespace kvb {

// Read-only view of host memory owned by something else (a mapped weight file).
struct TensorView {
  DType dtype{};
  std::vector<std::int64_t> shape;
  const std::byte* data = nullptr;
  std::size_t nbytes = 0;

  std::size_t numel() const noexcept { return nbytes / size_bytes(dtype); }

  // Elements as T. T must have the dtype's size and the data must be aligned for T.
  template <typename T>
  std::span<const T> as() const {
    if (sizeof(T) != size_bytes(dtype)) {
      throw Error("tensor element size does not match dtype " + std::string(dtype_name(dtype)));
    }
    if (reinterpret_cast<std::uintptr_t>(data) % alignof(T) != 0) {
      throw Error("tensor data is not aligned for its element type");
    }
    return {reinterpret_cast<const T*>(data), numel()};
  }
};

}  // namespace kvb
