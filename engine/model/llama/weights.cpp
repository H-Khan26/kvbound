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

DeviceWeights::DeviceWeights(const SafetensorsFile& file, std::span<const WeightParts> weights,
                             cudaStream_t stream)
    : format_(weight_format_from_metadata(file.metadata())) {
  // The arena starts with everything held as FP16 (floating-point tensors and scales), so one
  // pass can check that region for values that did not fit, and ends with the integer data.
  // The parts of a stacked weight sit back to back, with no padding between them, so the stack
  // reads as one row-major matrix (its scales likewise). A weight listed twice (tied embeddings)
  // shares one copy.
  struct Slot {
    Entry entry;
    const WeightParts* parts = nullptr;
  };
  const auto key_of = [](const WeightParts& parts) {
    std::string key;
    for (const std::string& part : parts) key += part + "\n";
    return key;
  };
  const auto scale_of = [&](const std::string& name) -> const TensorView& {
    const TensorView& scale = file.at(name + ".scale");
    if (scale.dtype != DType::F16) throw Error(name + ".scale: scales must be F16");
    return scale;
  };
  std::map<std::string, Slot> slots;
  std::size_t f16_bytes = 0, largest_f32 = 0;
  for (const WeightParts& parts : weights) {
    if (parts.empty()) throw Error("a weight needs at least one checkpoint tensor");
    std::string key = key_of(parts);
    if (slots.contains(key)) continue;
    Slot slot{.entry = {.scheme = scheme_of(parts[0], file.at(parts[0]), format_)},
              .parts = &parts};
    std::size_t bytes = 0;
    for (const std::string& name : parts) {
      const TensorView& tensor = file.at(name);
      if (scheme_of(name, tensor, format_) != slot.entry.scheme) {
        throw Error(name + ": stacked tensors must share one storage scheme");
      }
      if (slot.entry.scheme == WeightScheme::F16) {
        bytes += tensor.numel() * sizeof(std::uint16_t);
        if (tensor.dtype == DType::F32) largest_f32 = std::max(largest_f32, tensor.nbytes);
      } else {
        bytes += scale_of(name).nbytes;
      }
    }
    // FP16 data, or the scales of quantized data.
    (slot.entry.scheme == WeightScheme::F16 ? slot.entry.offset : slot.entry.scale_offset) =
        f16_bytes;
    f16_bytes += aligned(bytes);
    slots.emplace(std::move(key), slot);
  }
  std::size_t total = f16_bytes;
  for (auto& [key, slot] : slots) {
    if (slot.entry.scheme == WeightScheme::F16) continue;
    slot.entry.offset = total;
    std::size_t bytes = 0;
    for (const std::string& name : *slot.parts) bytes += file.at(name).nbytes;
    total += aligned(bytes);
  }
  for (const WeightParts& parts : weights) entries_.push_back(slots.at(key_of(parts)).entry);

  arena_ = DeviceBuffer(total);
  KVB_CUDA(cudaMemsetAsync(arena_.as(), 0, arena_.bytes(), stream));
  const DeviceBuffer f32_staging(largest_f32);
  std::byte* base = arena_.as<std::byte>();
  for (const auto& [key, slot] : slots) {
    std::byte* dst = base + slot.entry.offset;
    std::byte* scale_dst = base + slot.entry.scale_offset;
    for (const std::string& name : *slot.parts) {
      const TensorView& tensor = file.at(name);
      if (tensor.dtype == DType::F32) {
        KVB_CUDA(cudaMemcpyAsync(f32_staging.as(), tensor.data, tensor.nbytes,
                                 cudaMemcpyHostToDevice, stream));
        kernels::f32_to_f16(f32_staging.as<float>(), reinterpret_cast<std::uint16_t*>(dst),
                            tensor.numel(), stream);
        dst += tensor.numel() * sizeof(std::uint16_t);
        continue;
      }
      KVB_CUDA(cudaMemcpyAsync(dst, tensor.data, tensor.nbytes, cudaMemcpyHostToDevice, stream));
      if (tensor.dtype == DType::BF16) {
        kernels::bf16_to_f16(reinterpret_cast<std::uint16_t*>(dst), tensor.numel(), stream);
      }
      dst += tensor.nbytes;
      if (slot.entry.scheme != WeightScheme::F16) {
        const TensorView& scale = scale_of(name);
        KVB_CUDA(
            cudaMemcpyAsync(scale_dst, scale.data, scale.nbytes, cudaMemcpyHostToDevice, stream));
        scale_dst += scale.nbytes;
      }
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
