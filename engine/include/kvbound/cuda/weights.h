#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "kvbound/cuda/device.h"
#include "kvbound/safetensors.h"

namespace kvb::cuda {

// The model's weights in one device allocation, all FP16. Each tensor keeps the shape and the
// row-major order it has in the checkpoint.
class DeviceWeights {
 public:
  // Copies the named tensors from the mapped file to the device. BF16 and FP32 tensors are
  // converted to FP16 on the GPU. Throws kvb::Error if a tensor is missing, has another dtype,
  // or holds a value outside the FP16 range.
  DeviceWeights(const SafetensorsFile& file, std::span<const std::string> names,
                cudaStream_t stream);

  // Device pointer and shape of names[index].
  const std::uint16_t* data(int index) const noexcept {
    return arena_.as<std::uint16_t>() + entries_[index].offset;
  }
  const std::vector<std::int64_t>& shape(int index) const noexcept { return entries_[index].shape; }
  std::size_t bytes() const noexcept { return arena_.bytes(); }

 private:
  struct Entry {
    std::size_t offset = 0;  // in FP16 elements
    std::vector<std::int64_t> shape;
  };
  DeviceBuffer arena_;
  std::vector<Entry> entries_;
};

}  // namespace kvb::cuda
