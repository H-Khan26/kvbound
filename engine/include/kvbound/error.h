#pragma once

#include <stdexcept>

namespace kvb {

// Thrown for failures caused by input the engine does not control: malformed or truncated files,
// unsupported model variants.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace kvb
