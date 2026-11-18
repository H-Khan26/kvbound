#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "kvbound/cuda/blas.h"
#include "kvbound/cuda/device.h"
#include "kvbound/dtype.h"
#include "kvbound/weight_format.h"

namespace kvb::cuda {

// Which kernel multiplies a step's rows by a matrix.
struct LinearOptions {
  // FP16 matrices are multiplied by kvbound's own kernel for steps of up to this many rows, and
  // by cuBLAS for wider steps; 0 means always cuBLAS.
  int f16_gemv_rows = 4;
  // Quantized matrices are multiplied in place for steps of up to this many rows. Wider steps
  // use tensor cores on the quantized weights, or with expand_quantized, expand each matrix to
  // FP16 first for cuBLAS.
  int quantized_gemm_rows = 3;
  bool expand_quantized = false;
};

// Bytes of scratch run_linear() needs to expand a quantized matrix [n_out, n_in] about
// slice_bytes at a time: the whole matrix when it is smaller, and at least one output channel.
std::size_t expand_scratch_bytes(std::int64_t n_in, std::int64_t n_out, std::size_t slice_bytes);

// y [n_rows, n_out] = x [n_rows, n_in] * W^T + bias, with the kernel the options pick for the
// matrix's scheme and the step's rows. x and bias [n_out] are FP16; bias may be nullptr. y is
// FP16, or FP32 when out_dtype is F32. `blas` multiplies on its own stream, which must be
// `stream`. `dequant` is scratch for the expand_quantized path, of expand_scratch_bytes() or
// more: a quantized matrix is expanded and multiplied as many whole output channels at a time
// as fit in it.
void run_linear(const WeightRef& weight, const std::uint16_t* bias, const std::uint16_t* x, void* y,
                int n_rows, int n_in, int n_out, DType out_dtype, const LinearOptions& options,
                Blas& blas, const DeviceBuffer& dequant, cudaStream_t stream);

}  // namespace kvb::cuda
