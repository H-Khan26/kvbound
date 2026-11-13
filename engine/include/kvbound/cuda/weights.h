#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "kvbound/cuda/device.h"
#include "kvbound/safetensors.h"
#include "kvbound/weight_format.h"

namespace kvb::cuda {

// The model's weights in one device allocation. Floating-point tensors are held as FP16;
// quantized matrices keep the integers and FP16 scales of the checkpoint. Each tensor keeps the
// shape and the row-major order it has in the file.
class DeviceWeights {
 public:
  // Copies the named tensors from the mapped file to the device. BF16 and FP32 tensors are
  // converted to FP16 on the GPU; an integer tensor is taken as quantized under the file's
  // weight format, together with its "<name>.scale" tensor. Throws kvb::Error if a tensor is
  // missing, has a dtype the format does not explain, or holds a value outside the FP16 range.
  // Shapes are the caller's to check.
  DeviceWeights(const SafetensorsFile& file, std::span<const std::string> names,
                cudaStream_t stream);

  const WeightFormat& format() const noexcept { return format_; }
  // names[index] as the kernels receive it.
  WeightRef ref(int index) const noexcept;
  // Device pointer of names[index], which must be a floating-point tensor.
  const std::uint16_t* data(int index) const noexcept {
    return static_cast<const std::uint16_t*>(ref(index).data);
  }
  std::size_t bytes() const noexcept { return arena_.bytes(); }

 private:
  struct Entry {
    WeightScheme scheme = WeightScheme::F16;
    std::size_t offset = 0;        // of the tensor, in bytes
    std::size_t scale_offset = 0;  // of its scales, when quantized
  };

  WeightFormat format_;
  DeviceBuffer arena_;
  std::vector<Entry> entries_;  // per name
};

}  // namespace kvb::cuda
