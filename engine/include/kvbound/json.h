#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace kvb::json {

class Value;
using Array = std::vector<Value>;
// Members in file order. Objects here are small (config files, tensor headers), so lookup is a
// linear scan.
using Object = std::vector<std::pair<std::string, Value>>;

// One parsed JSON value. Accessors throw kvb::Error when the value has a different type.
class Value {
 public:
  using Storage =
      std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object>;

  Value() = default;
  explicit Value(Storage storage) : storage_(std::move(storage)) {}

  bool is_null() const noexcept { return std::holds_alternative<std::nullptr_t>(storage_); }

  bool as_bool() const;
  // Numbers written without a fraction or exponent stay exact 64-bit integers.
  std::int64_t as_int() const;
  double as_number() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  // Object member lookup. find() returns nullptr when the key is absent; at() throws.
  const Value* find(std::string_view key) const;
  const Value& at(std::string_view key) const;

 private:
  Storage storage_{nullptr};
};

// Parses one JSON document (RFC 8259). Throws kvb::Error with a byte offset on malformed input,
// duplicate object keys, or nesting deeper than 64 levels.
Value parse(std::string_view text);

}  // namespace kvb::json
