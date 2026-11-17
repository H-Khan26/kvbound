#include "kvbound/step.h"

#include "kvbound/error.h"

namespace kvb {

std::vector<AttnTile> plan_attention_tiles(std::span<const StepRow> rows, int max_rows) {
  if (max_rows < 1) throw Error("attention tiles need at least one row");
  std::vector<AttnTile> tiles;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    if (!tiles.empty()) {
      AttnTile& tile = tiles.back();
      const StepRow& prev = rows[r - 1];
      if (tile.n_rows < max_rows && rows[r].seq == prev.seq &&
          rows[r].position == prev.position + 1) {
        ++tile.n_rows;
        continue;
      }
    }
    tiles.push_back({.first_row = static_cast<std::int32_t>(r), .n_rows = 1});
  }
  return tiles;
}

}  // namespace kvb
