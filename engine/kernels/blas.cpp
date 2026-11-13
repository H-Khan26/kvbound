#include "kvbound/cuda/blas.h"

#include <cublas_v2.h>

#include <string>

#include "kvbound/error.h"

namespace kvb::cuda {
namespace {

void check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw Error(std::string(what) + ": cuBLAS status " + std::to_string(status));
  }
}

}  // namespace

std::string_view gemm_accumulate_name(GemmAccumulate mode) noexcept {
  return mode == GemmAccumulate::F16 ? "fp16" : "fp32";
}

std::optional<GemmAccumulate> gemm_accumulate_from_name(std::string_view name) noexcept {
  if (name == "fp16") return GemmAccumulate::F16;
  if (name == "fp32") return GemmAccumulate::F32;
  return std::nullopt;
}

Blas::Blas(cudaStream_t stream, GemmAccumulate accumulate) : accumulate_(accumulate) {
  check(cublasCreate(&handle_), "cublasCreate");
  check(cublasSetStream(handle_, stream), "cublasSetStream");
}

Blas::~Blas() {
  if (handle_ != nullptr) cublasDestroy(handle_);
}

void Blas::linear(const std::uint16_t* weight, const std::uint16_t* x, void* y, int n_rows,
                  int n_in, int n_out, DType out_dtype, int y_cols) {
  // cuBLAS is column-major. A row-major [rows, cols] matrix is its own transpose read
  // column-major, so with W' = weight (n_in x n_out) and X' = x (n_in x n_rows) the product
  // y' (n_out x n_rows) = W'^T * X' is y in row-major order.
  const bool f32_out = out_dtype == DType::F32;
  const bool f32_sums = f32_out || accumulate_ == GemmAccumulate::F32;
  const float one_f32 = 1.0f, zero_f32 = 0.0f;
  const std::uint16_t one_f16 = 0x3C00, zero_f16 = 0;
  const void* alpha = f32_sums ? static_cast<const void*>(&one_f32) : &one_f16;
  const void* beta = f32_sums ? static_cast<const void*>(&zero_f32) : &zero_f16;
  check(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n_out, n_rows, n_in, alpha, weight,
                     CUDA_R_16F, n_in, x, CUDA_R_16F, n_in, beta, y,
                     f32_out ? CUDA_R_32F : CUDA_R_16F, y_cols > 0 ? y_cols : n_out,
                     f32_sums ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

}  // namespace kvb::cuda
