#include "kvbound/error.h"

#include <cstdio>
#include <cstdlib>

namespace kvb {

void check_failed(const char* expr, const char* file, int line) {
  std::fprintf(stderr, "KVB_CHECK failed: %s at %s:%d\n", expr, file, line);
  std::abort();
}

}  // namespace kvb
