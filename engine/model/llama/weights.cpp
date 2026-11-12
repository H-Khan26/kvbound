#include "kvbound/cuda/weights.h"

#include <algorithm>
#include <map>

#include "kvbound/cuda/kernels.h"
#include "kvbound/memory_plan.h"

namespace kvb::cuda {

DeviceWeights::DeviceWeights(const SafetensorsFile& file, std::span<const std::string> names,
                             cudaStream_t stream) {
  constexpr std::size_t kAlignElements = kArenaAlign / sizeof(std::uint16_t);

  // A name listed twice (tied embeddings) shares one copy.
  std::map<std::string_view, std::size_t> offsets;
  std::size_t total = 0, largest_f32 = 0;
  for (const std::string& name : names) {
    const TensorView& tensor = file.at(name);
    if (tensor.dtype != DType::F16 && tensor.dtype != DType::BF16 && tensor.dtype != DType::F32) {
      throw Error(name + ": weights must be F16, BF16 or F32");
    }
    const auto [it, inserted] = offsets.try_emplace(name, total);
    if (inserted) {
      total += (tensor.numel() + kAlignElements - 1) / kAlignElements * kAlignElements;
      if (tensor.dtype == DType::F32) largest_f32 = std::max(largest_f32, tensor.nbytes);
    }
    entries_.push_back({it->second, tensor.shape});
  }

  arena_ = DeviceBuffer(total * sizeof(std::uint16_t));
  KVB_CUDA(cudaMemsetAsync(arena_.as(), 0, arena_.bytes(), stream));
  const DeviceBuffer f32_staging(largest_f32);
  for (const auto& [name, offset] : offsets) {
    const TensorView& tensor = file.at(name);
    std::uint16_t* dst = arena_.as<std::uint16_t>() + offset;
    if (tensor.dtype == DType::F32) {
      KVB_CUDA(cudaMemcpyAsync(f32_staging.as(), tensor.data, tensor.nbytes, cudaMemcpyHostToDevice,
                               stream));
      kernels::f32_to_f16(f32_staging.as<float>(), dst, tensor.numel(), stream);
    } else {
      KVB_CUDA(cudaMemcpyAsync(dst, tensor.data, tensor.nbytes, cudaMemcpyHostToDevice, stream));
      if (tensor.dtype == DType::BF16) kernels::bf16_to_f16(dst, tensor.numel(), stream);
    }
  }
  if (kernels::count_non_finite(arena_.as<std::uint16_t>(), total, stream) != 0) {
    throw Error("weights hold values that are not finite in FP16");
  }
}

}  // namespace kvb::cuda
