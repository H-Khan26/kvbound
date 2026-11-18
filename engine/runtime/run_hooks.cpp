#include "kvbound/cuda/run_hooks.h"

#include <algorithm>

#include "kvbound/cuda/device.h"
#include "kvbound/dtype.h"

namespace kvb::cuda {

RunHooks step_hooks(const Program& program, int n_tokens, int n_out, cudaStream_t stream,
                    Capture* capture, OpTimes* times) {
  RunHooks hooks;
  hooks.times = times;
  if (capture == nullptr) return hooks;
  hooks.on_site = [&program, n_tokens, n_out, stream, capture](const std::string& site, int id,
                                                               const void* data) {
    const Value& value = program.values[id];
    const std::size_t n =
        static_cast<std::size_t>(row_count(value.rows, n_tokens, n_out) * value.cols);
    std::vector<float>& host = (*capture)[site];
    host.resize(n);
    if (value.dtype == DType::F32) {
      KVB_CUDA(
          cudaMemcpyAsync(host.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
      KVB_CUDA(cudaStreamSynchronize(stream));
      return;
    }
    std::vector<std::uint16_t> bits(n);
    KVB_CUDA(cudaMemcpyAsync(bits.data(), data, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                             stream));
    KVB_CUDA(cudaStreamSynchronize(stream));
    std::transform(bits.begin(), bits.end(), host.begin(), half_bits_to_float);
  };
  return hooks;
}

}  // namespace kvb::cuda
