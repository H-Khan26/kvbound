#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <string_view>

#include "kvbound/dtype.h"

struct cublasContext;

namespace kvb::cuda {

// Precision of the running sums inside an FP16 matrix product. FP32 is exact to more digits;
// FP16 is the faster tensor-core mode.
enum class GemmAccumulate : std::uint8_t { F16, F32 };

std::string_view gemm_accumulate_name(GemmAccumulate mode) noexcept;
std::optional<GemmAccumulate> gemm_accumulate_from_name(std::string_view name) noexcept;

// The engine's matrix products, on cuBLAS. Move-only owner of one cuBLAS handle.
class Blas {
 public:
  Blas(cudaStream_t stream, GemmAccumulate accumulate);
  ~Blas();
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;

  // y [n_rows, n_out] = x [n_rows, n_in] * weight^T, with weight [n_out, n_in] as stored in the
  // checkpoint. x and weight are FP16; y is FP16, or FP32 when out_dtype is F32 (which always
  // accumulates in FP32).
  void linear(const std::uint16_t* weight, const std::uint16_t* x, void* y, int n_rows, int n_in,
              int n_out, DType out_dtype = DType::F16);

  GemmAccumulate accumulate() const noexcept { return accumulate_; }

 private:
  cublasContext* handle_ = nullptr;
  GemmAccumulate accumulate_;
};

}  // namespace kvb::cuda
