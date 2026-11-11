#pragma once

#include <stdexcept>
#include <string>

namespace kvb {

// Thrown for failures caused by input the engine does not control: malformed or truncated files,
// unsupported model variants. Broken internal invariants use KVB_CHECK instead.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void check_failed(const char* expr, const char* file, int line);

}  // namespace kvb

// Aborts with file and line when an internal invariant does not hold. Active in every build type.
#define KVB_CHECK(cond) \
  ((cond) ? static_cast<void>(0) : ::kvb::check_failed(#cond, __FILE__, __LINE__))
