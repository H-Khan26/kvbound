#include "kvbound/kv_layout.h"

#include <array>
#include <utility>

namespace kvb {
namespace {

constexpr std::array<std::pair<KvLayout, std::string_view>, 3> kNames{{
    {KvLayout::HeadMajor, "head_major"},
    {KvLayout::TokenMajor, "token_major"},
    {KvLayout::DimMajor, "dim_major"},
}};

}  // namespace

std::string_view kv_layout_name(KvLayout layout) noexcept {
  for (const auto& [value, name] : kNames) {
    if (value == layout) return name;
  }
  return "?";
}

std::optional<KvLayout> kv_layout_from_name(std::string_view name) noexcept {
  for (const auto& [value, known] : kNames) {
    if (known == name) return value;
  }
  return std::nullopt;
}

KvStrides kv_strides(KvLayout layout, const KvDims& d) noexcept {
  KvStrides s;
  switch (layout) {
    case KvLayout::HeadMajor:
      s.dim = 1;
      s.pos = d.head_dim;
      s.head = s.pos * d.block_size;
      s.block = s.head * d.n_kv_heads;
      s.layer = s.block * d.n_blocks;
      break;
    case KvLayout::TokenMajor:
      s.dim = 1;
      s.head = d.head_dim;
      s.layer = s.head * d.n_kv_heads;
      s.pos = s.layer * d.n_layers;
      s.block = s.pos * d.block_size;
      break;
    case KvLayout::DimMajor:
      s.pos = 1;
      s.dim = d.block_size;
      s.head = s.dim * d.head_dim;
      s.block = s.head * d.n_kv_heads;
      s.layer = s.block * d.n_blocks;
      break;
  }
  return s;
}

}  // namespace kvb
