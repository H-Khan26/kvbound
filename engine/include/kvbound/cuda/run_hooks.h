#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "kvbound/program.h"

namespace kvb::cuda {

// Wall time spent in each kind of op, measured by synchronizing after every launch.
struct OpTimes {
  std::array<double, kOpKindCount> seconds{};
  std::array<std::int64_t, kOpKindCount> calls{};
};

// What a model reports while it runs a program.
struct RunHooks {
  // Called for each capture site an op fills, with the program value and its device pointer.
  std::function<void(const std::string& site, int value, const void* data)> on_site;
  // When set, the stream is synchronized after every op and the time is added here.
  OpTimes* times = nullptr;
};

// Capture-site name to values, row-major with the rows of the site's value, as FP32.
using Capture = std::map<std::string, std::vector<float>>;

// Hooks for one step of `program` with n_tokens rows and n_out outputs: every site is copied
// from `stream` into `capture` as it is filled (nullptr captures nothing), and ops are timed
// into `times` (nullptr times nothing). The program and the capture must outlive the hooks.
RunHooks step_hooks(const Program& program, int n_tokens, int n_out, cudaStream_t stream,
                    Capture* capture, OpTimes* times);

}  // namespace kvb::cuda
