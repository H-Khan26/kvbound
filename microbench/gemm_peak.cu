// Achievable matrix-multiply throughput per precision, measured with cuBLASLt.
//
// Part 1 (peak): large square GEMMs per number format. These are the compute ceilings of the
// roofline: the most arithmetic per second this card delivers when memory is not the limit.
//
// Part 2 (shapes): FP16 GEMMs shaped like one linear layer of the model, W[rows x cols] times a
// batch of `tokens` activation vectors. One token is a decode step (every weight is read to do
// two flops with it, so memory bandwidth limits it); hundreds of tokens is prefill (each weight
// is reused per token, so compute limits it). The calls cycle through enough distinct weight
// matrices to exceed any cache, as a real decode step does when it walks the whole model.

#include <cublasLt.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "common.h"

namespace {

constexpr std::size_t kWorkspaceBytes = 64u << 20;
constexpr double kTargetSeconds = 0.25;
constexpr int kRepeats = 5;

const char* status_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "not_supported";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "arch_mismatch";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "invalid_value";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "alloc_failed";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "execution_failed";
    default:
      return "error";
  }
}

std::size_t element_bytes(cudaDataType_t type) {
  switch (type) {
    case CUDA_R_32F:
    case CUDA_R_32I:
      return 4;
    case CUDA_R_16F:
      return 2;
    default:
      return 1;  // CUDA_R_8I, CUDA_R_8F_E4M3
  }
}

// One number format: input and output element types plus the type arithmetic accumulates in.
struct Precision {
  const char* name;
  cudaDataType_t input;
  cudaDataType_t output;
  cublasComputeType_t compute;
  cudaDataType_t scale;        // type of alpha and beta
  int min_compute_capability;  // major * 10 + minor
};

constexpr Precision kPrecisions[] = {
    {"fp32", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F, 0},
    {"fp16_acc16", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F, 0},
    {"fp16_acc32", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, 0},
    {"int8", CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, 0},
    {"fp8_e4m3", CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, 89},
};

// alpha = 1 and beta = 0 encoded in the scale type.
struct Scalars {
  std::uint8_t one[4];
  std::uint8_t zero[4] = {0, 0, 0, 0};
  explicit Scalars(cudaDataType_t type) {
    const float f = 1.0f;
    const std::int32_t i = 1;
    const std::uint16_t h = 0x3C00;  // 1.0 as a half
    std::memset(one, 0, sizeof(one));
    if (type == CUDA_R_32F) std::memcpy(one, &f, 4);
    if (type == CUDA_R_32I) std::memcpy(one, &i, 4);
    if (type == CUDA_R_16F) std::memcpy(one, &h, 2);
  }
};

// A planned C[m x n] = A[m x k] * B[k x n] in column-major order, ready to run repeatedly.
class Gemm {
 public:
  Gemm(cublasLtHandle_t handle, const Precision& p, std::uint64_t m, std::uint64_t n,
       std::uint64_t k)
      : handle_(handle), scalars_(p.scale) {
    status_ = cublasLtMatmulDescCreate(&desc_, p.compute, p.scale);
    if (ok()) status_ = cublasLtMatrixLayoutCreate(&a_, p.input, m, k, m);
    if (ok()) status_ = cublasLtMatrixLayoutCreate(&b_, p.input, k, n, k);
    if (ok()) status_ = cublasLtMatrixLayoutCreate(&c_, p.output, m, n, m);
    if (ok()) status_ = cublasLtMatmulPreferenceCreate(&pref_);
    if (ok()) {
      status_ =
          cublasLtMatmulPreferenceSetAttribute(pref_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes));
    }
    if (ok()) {
      int found = 0;
      status_ = cublasLtMatmulAlgoGetHeuristic(handle_, desc_, a_, b_, c_, c_, pref_, 1,
                                               &heuristic_, &found);
      if (ok() && found == 0) status_ = CUBLAS_STATUS_NOT_SUPPORTED;
      if (ok()) status_ = heuristic_.state;
    }
  }

  ~Gemm() {
    if (pref_) cublasLtMatmulPreferenceDestroy(pref_);
    if (c_) cublasLtMatrixLayoutDestroy(c_);
    if (b_) cublasLtMatrixLayoutDestroy(b_);
    if (a_) cublasLtMatrixLayoutDestroy(a_);
    if (desc_) cublasLtMatmulDescDestroy(desc_);
  }
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;

  bool ok() const { return status_ == CUBLAS_STATUS_SUCCESS; }
  cublasStatus_t status() const { return status_; }

  void run(const void* a, const void* b, void* c, void* workspace) {
    const cublasStatus_t s =
        cublasLtMatmul(handle_, desc_, scalars_.one, a, a_, b, b_, scalars_.zero, c, c_, c, c_,
                       &heuristic_.algo, workspace, kWorkspaceBytes, nullptr);
    if (s != CUBLAS_STATUS_SUCCESS) status_ = s;
  }

 private:
  cublasLtHandle_t handle_;
  Scalars scalars_;
  cublasStatus_t status_ = CUBLAS_STATUS_SUCCESS;
  cublasLtMatmulDesc_t desc_ = nullptr;
  cublasLtMatrixLayout_t a_ = nullptr, b_ = nullptr, c_ = nullptr;
  cublasLtMatmulPreference_t pref_ = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic_{};
};

struct Timing {
  cublasStatus_t status;
  double best_seconds;    // per call
  double median_seconds;  // per call
};

// Times `call(i)` for i = 0, 1, 2, ... in batches sized to run about kTargetSeconds.
template <typename Call>
Timing time_calls(Gemm& gemm, Call&& call) {
  call(0);  // warm-up; also surfaces unsupported configurations
  CUDA_CHECK(cudaDeviceSynchronize());
  if (!gemm.ok()) return {gemm.status(), 0, 0};

  int calls = 1;
  double seconds = kvb::bench::time_gpu([&] { call(0); });
  while (seconds < kTargetSeconds / 4 && calls < (1 << 20)) {
    calls *= 4;
    seconds = kvb::bench::time_gpu([&] {
      for (int i = 0; i < calls; ++i) call(i);
    });
  }
  calls = std::max(1, static_cast<int>(calls * kTargetSeconds / seconds));

  std::vector<double> per_call;
  for (int r = 0; r < kRepeats; ++r) {
    const double t = kvb::bench::time_gpu([&] {
      for (int i = 0; i < calls; ++i) call(i);
    });
    per_call.push_back(t / calls);
  }
  return {gemm.status(), *std::min_element(per_call.begin(), per_call.end()),
          kvb::bench::median(per_call)};
}

}  // namespace

int main() {
  const cudaDeviceProp prop = kvb::bench::device_properties();
  const int compute_capability = prop.major * 10 + prop.minor;

  cublasLtHandle_t handle = nullptr;
  if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasLtCreate failed\n");
    return 1;
  }

  // One pool reused by every measurement. Zeroed memory is a valid operand in every format, and
  // GEMM timing does not depend on the values.
  constexpr std::size_t kPoolBytes = 1u << 30;
  constexpr std::size_t kOutBytes = 512u << 20;
  void *pool = nullptr, *out = nullptr, *workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&pool, kPoolBytes));
  CUDA_CHECK(cudaMalloc(&out, kOutBytes));
  CUDA_CHECK(cudaMalloc(&workspace, kWorkspaceBytes));
  CUDA_CHECK(cudaMemset(pool, 0, kPoolBytes));
  CUDA_CHECK(cudaMemset(out, 0, kOutBytes));

  std::printf("{\n  \"benchmark\": \"gemm_peak\",\n");
  kvb::bench::print_device_json(prop);
  std::printf(
      "  \"params\": {\"library\": \"cublasLt\", \"target_seconds\": %.2f, "
      "\"repeats\": %d},\n  \"peak\": [",
      kTargetSeconds, kRepeats);

  // Part 1: square GEMMs. A and B are the first and second halves of the pool.
  bool first = true;
  for (const Precision& p : kPrecisions) {
    for (const std::uint64_t n : {2048ull, 4096ull, 8192ull}) {
      const char* sep = first ? "" : ",";
      first = false;
      std::printf("%s\n    {\"precision\": \"%s\", \"n\": %llu, ", sep, p.name, n);
      if (compute_capability < p.min_compute_capability) {
        std::printf("\"status\": \"needs_sm_%d\"}", p.min_compute_capability);
        continue;
      }
      Gemm gemm(handle, p, n, n, n);
      Timing t{gemm.status(), 0, 0};
      if (gemm.ok()) {
        const void* b = static_cast<const char*>(pool) + kPoolBytes / 2;
        t = time_calls(gemm, [&](int) { gemm.run(pool, b, out, workspace); });
      }
      if (t.status != CUBLAS_STATUS_SUCCESS) {
        std::printf("\"status\": \"%s\"}", status_name(t.status));
        continue;
      }
      const double ops = 2.0 * n * n * n;
      std::printf("\"status\": \"success\", \"best_tops\": %.2f, \"median_tops\": %.2f}",
                  ops / t.best_seconds / 1e12, ops / t.median_seconds / 1e12);
    }
  }

  // Part 2: one FP16 linear layer applied to a batch of token vectors.
  const Precision& fp16 = kPrecisions[2];  // FP32 accumulate, the accuracy-safe configuration
  const std::uint64_t rows = 2048, cols = 2048;
  const std::size_t weight_bytes = rows * cols * element_bytes(fp16.input);
  const std::size_t x_reserve = 1024 * cols * element_bytes(fp16.input);
  const int n_weights = static_cast<int>((kPoolBytes - x_reserve) / weight_bytes);
  const char* x = static_cast<const char*>(pool) + kPoolBytes - x_reserve;

  std::printf(
      "\n  ],\n  \"shapes\": {\"precision\": \"%s\", \"rows\": %llu, \"cols\": %llu, "
      "\"distinct_weight_matrices\": %d, \"points\": [",
      fp16.name, rows, cols, n_weights);
  first = true;
  for (std::uint64_t tokens = 1; tokens <= 1024; tokens *= 2) {
    Gemm gemm(handle, fp16, rows, tokens, cols);
    Timing t{gemm.status(), 0, 0};
    if (gemm.ok()) {
      t = time_calls(gemm, [&](int i) {
        const char* w = static_cast<const char*>(pool) + (i % n_weights) * weight_bytes;
        gemm.run(w, x, out, workspace);
      });
    }
    std::printf("%s\n      {\"tokens\": %llu, ", first ? "" : ",", tokens);
    first = false;
    if (t.status != CUBLAS_STATUS_SUCCESS) {
      std::printf("\"status\": \"%s\"}", status_name(t.status));
      continue;
    }
    const double ops = 2.0 * rows * cols * tokens;
    std::printf(
        "\"status\": \"success\", \"best_gflops\": %.2f, \"median_gflops\": %.2f, "
        "\"weight_gbs\": %.2f}",
        ops / t.best_seconds / 1e9, ops / t.median_seconds / 1e9,
        weight_bytes / t.best_seconds / 1e9);
  }
  std::printf("\n    ]\n  }\n}\n");

  CUDA_CHECK(cudaFree(pool));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(workspace));
  cublasLtDestroy(handle);
  return 0;
}
