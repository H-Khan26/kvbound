#include "kvbound/cuda/linear.h"

#include <algorithm>
#include <cstddef>

#include "kvbound/cuda/kernels.h"

namespace kvb::cuda {

std::size_t expand_scratch_bytes(std::int64_t n_in, std::int64_t n_out, std::size_t slice_bytes) {
  const std::size_t row_bytes = static_cast<std::size_t>(n_in) * sizeof(std::uint16_t);
  const std::size_t matrix = static_cast<std::size_t>(n_out) * row_bytes;
  return std::min(matrix, std::max(slice_bytes, row_bytes));
}

void run_linear(const WeightRef& weight, const std::uint16_t* bias, const std::uint16_t* x, void* y,
                int n_rows, int n_in, int n_out, DType out_dtype, const LinearOptions& options,
                Blas& blas, const DeviceBuffer& dequant, cudaStream_t stream) {
  using kernels::f16;
  const bool f32_out = out_dtype == DType::F32;
  if (weight.scheme == WeightScheme::F16) {
    const f16* matrix = static_cast<const f16*>(weight.data);
    if (n_rows <= options.f16_gemv_rows && n_in % 8 == 0) {
      kernels::linear_f16(matrix, bias, x, y, n_rows, n_in, n_out, f32_out, stream);
      return;
    }
    blas.linear(matrix, x, y, n_rows, n_in, n_out, out_dtype);
  } else if (n_rows <= options.quantized_gemm_rows) {
    kernels::linear_quantized(weight, bias, x, y, n_rows, n_in, n_out, f32_out, stream);
    return;
  } else if (!options.expand_quantized) {
    kernels::gemm_quantized(weight, bias, x, y, n_rows, n_in, n_out, f32_out, stream);
    return;
  } else {
    // Expanded and multiplied a slice of output channels at a time.
    const int slice = static_cast<int>(dequant.bytes() / sizeof(f16) / n_in);
    if (slice < 1) throw Error("no scratch to expand a quantized matrix into");
    const std::size_t element = size_bytes(out_dtype);
    for (int first = 0; first < n_out; first += slice) {
      const int count = std::min(slice, n_out - first);
      kernels::dequantize(weight, dequant.as<f16>(), n_in, first, count, stream);
      blas.linear(dequant.as<f16>(), x, static_cast<std::byte*>(y) + first * element, n_rows, n_in,
                  count, out_dtype, n_out);
    }
  }
  // The cuBLAS products add no bias of their own.
  if (bias != nullptr) kernels::add_bias(y, bias, n_rows, n_out, f32_out, stream);
}

}  // namespace kvb::cuda
