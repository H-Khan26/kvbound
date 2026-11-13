#include "kvbound/cuda/weights.h"

#include <algorithm>
#include <map>

#include "kvbound/cuda/kernels.h"
#include "kvbound/memory_plan.h"

namespace kvb::cuda {
namespace {

std::size_t aligned(std::size_t bytes) {
  return (bytes + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
}

WeightScheme scheme_of(const std::string& name, const TensorView& tensor,
                       const WeightFormat& format) {
  switch (tensor.dtype) {
    case DType::F16:
    case DType::BF16:
    case DType::F32:
      return WeightScheme::F16;
    case DType::I8:
      if (format.scheme == WeightScheme::Int8PerChannel) return format.scheme;
      break;
    case DType::U8:
      if (format.scheme == WeightScheme::Int4Grouped) return format.scheme;
      break;
    default:
      break;
  }
  throw Error(name + ": dtype " + std::string(dtype_name(tensor.dtype)) +
              " does not fit weight format " + std::string(weight_scheme_name(format.scheme)));
}

}  // namespace

DeviceWeights::DeviceWeights(const SafetensorsFile& file, std::span<const std::string> names,
                             cudaStream_t stream)
    : format_(weight_format_from_metadata(file.metadata())) {
  // The arena starts with everything held as FP16 (floating-point tensors and scales), so one
  // pass can check that region for values that did not fit, and ends with the integer data.
  // A name listed twice (tied embeddings) shares one copy.
  struct Slot {
    Entry entry;
    const TensorView* scale = nullptr;
  };
  std::map<std::string_view, Slot> slots;
  std::size_t f16_bytes = 0, largest_f32 = 0;
  for (const std::string& name : names) {
    if (slots.contains(name)) continue;
    const TensorView& tensor = file.at(name);
    Slot slot{.entry = {.scheme = scheme_of(name, tensor, format_)}};
    if (slot.entry.scheme == WeightScheme::F16) {
      slot.entry.offset = f16_bytes;
      f16_bytes += aligned(tensor.numel() * sizeof(std::uint16_t));
      if (tensor.dtype == DType::F32) largest_f32 = std::max(largest_f32, tensor.nbytes);
    } else {
      slot.scale = &file.at(name + ".scale");
      if (slot.scale->dtype != DType::F16) throw Error(name + ".scale: scales must be F16");
      slot.entry.scale_offset = f16_bytes;
      f16_bytes += aligned(slot.scale->nbytes);
    }
    slots.emplace(name, slot);
  }
  std::size_t total = f16_bytes;
  for (auto& [name, slot] : slots) {
    if (slot.entry.scheme == WeightScheme::F16) continue;
    slot.entry.offset = total;
    total += aligned(file.at(name).nbytes);
  }
  for (const std::string& name : names) entries_.push_back(slots.at(name).entry);

  arena_ = DeviceBuffer(total);
  KVB_CUDA(cudaMemsetAsync(arena_.as(), 0, arena_.bytes(), stream));
  const DeviceBuffer f32_staging(largest_f32);
  std::byte* base = arena_.as<std::byte>();
  for (const auto& [name, slot] : slots) {
    const TensorView& tensor = file.at(name);
    std::byte* dst = base + slot.entry.offset;
    if (tensor.dtype == DType::F32) {
      KVB_CUDA(cudaMemcpyAsync(f32_staging.as(), tensor.data, tensor.nbytes, cudaMemcpyHostToDevice,
                               stream));
      kernels::f32_to_f16(f32_staging.as<float>(), reinterpret_cast<std::uint16_t*>(dst),
                          tensor.numel(), stream);
      continue;
    }
    KVB_CUDA(cudaMemcpyAsync(dst, tensor.data, tensor.nbytes, cudaMemcpyHostToDevice, stream));
    if (tensor.dtype == DType::BF16) {
      kernels::bf16_to_f16(reinterpret_cast<std::uint16_t*>(dst), tensor.numel(), stream);
    }
    if (slot.scale != nullptr) {
      KVB_CUDA(cudaMemcpyAsync(base + slot.entry.scale_offset, slot.scale->data, slot.scale->nbytes,
                               cudaMemcpyHostToDevice, stream));
    }
  }
  if (kernels::count_non_finite(arena_.as<std::uint16_t>(), f16_bytes / sizeof(std::uint16_t),
                                stream) != 0) {
    throw Error("weights hold values that are not finite in FP16");
  }
}

WeightRef DeviceWeights::ref(int index) const noexcept {
  const Entry& entry = entries_[index];
  const std::byte* base = arena_.as<std::byte>();
  return WeightRef{
      .scheme = entry.scheme,
      .data = base + entry.offset,
      .scale = entry.scheme == WeightScheme::F16
                   ? nullptr
                   : reinterpret_cast<const std::uint16_t*>(base + entry.scale_offset),
      .group_size = format_.group_size,
  };
}

}  // namespace kvb::cuda
