#include "kvbound/cuda/step_buffers.h"

#include <cstring>
#include <string>

namespace kvb::cuda {
namespace {

// Copies `values` into the pinned buffer and queues the copy from there to the device.
template <typename T>
void upload(std::span<const T> values, const PinnedBuffer& host, const DeviceBuffer& device,
            cudaStream_t stream) {
  if (values.size_bytes() > host.bytes()) throw Error("step exceeds its buffers");
  if (values.empty()) return;
  std::memcpy(host.as(), values.data(), values.size_bytes());
  KVB_CUDA(
      cudaMemcpyAsync(device.as(), host.as(), values.size_bytes(), cudaMemcpyHostToDevice, stream));
}

}  // namespace

void check_step(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
                const StepLimits& limits) {
  const int n_tokens = static_cast<int>(rows.size());
  const int n_out = static_cast<int>(out_rows.size());
  if (n_tokens == 0 || n_tokens > limits.max_step_tokens || n_out > n_tokens ||
      n_out > limits.max_outputs) {
    throw Error("step has " + std::to_string(n_tokens) + " rows and " + std::to_string(n_out) +
                " outputs; the limits are " + std::to_string(limits.max_step_tokens) + " and " +
                std::to_string(limits.max_outputs));
  }
  for (const StepRow& row : rows) {
    if (row.position < 0 || row.position >= limits.max_ctx || row.seq < 0 ||
        row.seq >= limits.max_seqs || row.token_id < 0 || row.token_id >= limits.vocab_size) {
      throw Error("step row is out of range");
    }
  }
  for (const std::int32_t row : out_rows) {
    if (row < 0 || row >= n_tokens) throw Error("output row is out of range");
  }
}

StepBuffers::StepBuffers(std::size_t max_rows, std::size_t max_outputs)
    : rows_host_(max_rows * sizeof(StepRow)),
      out_rows_host_(max_outputs * sizeof(std::int32_t)),
      tiles_host_(max_rows * sizeof(AttnTile)),
      next_ids_(max_outputs * sizeof(std::int32_t)),
      rows_dev_(rows_host_.bytes()),
      out_rows_dev_(out_rows_host_.bytes()),
      tiles_dev_(tiles_host_.bytes()) {}

void StepBuffers::upload_rows(std::span<const StepRow> rows, cudaStream_t stream) {
  upload(rows, rows_host_, rows_dev_, stream);
}

void StepBuffers::upload_out_rows(std::span<const std::int32_t> out_rows, cudaStream_t stream) {
  upload(out_rows, out_rows_host_, out_rows_dev_, stream);
}

void StepBuffers::upload_tiles(std::span<const AttnTile> tiles, cudaStream_t stream) {
  upload(tiles, tiles_host_, tiles_dev_, stream);
}

void StepBuffers::fetch_next_ids(const std::int32_t* ids, std::size_t n_out, cudaStream_t stream) {
  if (n_out * sizeof(std::int32_t) > next_ids_.bytes()) throw Error("step exceeds its buffers");
  if (n_out > 0) {
    KVB_CUDA(cudaMemcpyAsync(next_ids_.as(), ids, n_out * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost, stream));
  }
  n_next_ids_ = n_out;
}

}  // namespace kvb::cuda
