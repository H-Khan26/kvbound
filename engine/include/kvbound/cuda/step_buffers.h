#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

#include "kvbound/cuda/device.h"
#include "kvbound/step.h"

namespace kvb::cuda {

// What one step of token rows may hold.
struct StepLimits {
  int max_step_tokens = 0;
  int max_outputs = 0;
  int max_ctx = 0;   // positions are below this
  int max_seqs = 0;  // block-table rows are below this
  std::int64_t vocab_size = 0;
};

// Throws kvb::Error unless the step has between one row and max_step_tokens, at most
// max_outputs output rows, each naming a row of the step, and every row within the limits.
void check_step(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
                const StepLimits& limits);

// The inputs and outputs of a step: pinned host copies and their persistent device buffers. The
// device addresses never change, so a captured CUDA graph can keep reading them.
class StepBuffers {
 public:
  StepBuffers() = default;
  // Room for steps of max_rows rows, of which max_outputs produce a token.
  StepBuffers(std::size_t max_rows, std::size_t max_outputs);

  // Each upload copies into the pinned buffer and queues its copy to the device on `stream`.
  // The pinned buffer is read when the stream gets there, so the caller synchronizes the stream
  // before it uploads the next step.
  void upload_rows(std::span<const StepRow> rows, cudaStream_t stream);
  void upload_out_rows(std::span<const std::int32_t> out_rows, cudaStream_t stream);
  void upload_tiles(std::span<const AttnTile> tiles, cudaStream_t stream);
  // Queues the copy of a step's n_out greedy tokens from the device array `ids` to the host.
  void fetch_next_ids(const std::int32_t* ids, std::size_t n_out, cudaStream_t stream);

  const StepRow* rows() const noexcept { return rows_dev_.as<StepRow>(); }
  const std::int32_t* out_rows() const noexcept { return out_rows_dev_.as<std::int32_t>(); }
  const AttnTile* tiles() const noexcept { return tiles_dev_.as<AttnTile>(); }
  // The tokens of the latest fetch, valid once the stream is synchronized.
  std::span<const std::int32_t> next_ids() const noexcept {
    return {next_ids_.as<std::int32_t>(), n_next_ids_};
  }

 private:
  PinnedBuffer rows_host_, out_rows_host_, tiles_host_, next_ids_;
  DeviceBuffer rows_dev_, out_rows_dev_, tiles_dev_;
  std::size_t n_next_ids_ = 0;
};

}  // namespace kvb::cuda
