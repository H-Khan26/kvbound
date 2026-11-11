#include "kvbound/json.h"

#include <charconv>

#include "kvbound/error.h"

namespace kvb::json {
namespace {

constexpr int kMaxDepth = 64;

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  Value parse_document() {
    Value value = parse_value(0);
    skip_whitespace();
    if (pos_ != text_.size()) fail("unexpected content after the document");
    return value;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    throw Error("json: " + std::string(what) + " at byte " + std::to_string(pos_));
  }

  void skip_whitespace() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      ++pos_;
    }
  }

  char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++pos_;
  }

  bool consume_literal(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) != literal) return false;
    pos_ += literal.size();
    return true;
  }

  Value parse_value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    skip_whitespace();
    switch (peek()) {
      case '{':
        return parse_object(depth);
      case '[':
        return parse_array(depth);
      case '"':
        return Value(Value::Storage(parse_string()));
      case 't':
        if (consume_literal("true")) return Value(Value::Storage(true));
        break;
      case 'f':
        if (consume_literal("false")) return Value(Value::Storage(false));
        break;
      case 'n':
        if (consume_literal("null")) return Value();
        break;
      default:
        return parse_number();
    }
    fail("invalid literal");
  }

  Value parse_object(int depth) {
    expect('{');
    Object members;
    skip_whitespace();
    if (peek() == '}') {
      ++pos_;
      return Value(Value::Storage(std::move(members)));
    }
    while (true) {
      skip_whitespace();
      if (peek() != '"') fail("expected a string key");
      std::string key = parse_string();
      for (const auto& member : members) {
        if (member.first == key) fail("duplicate object key");
      }
      skip_whitespace();
      expect(':');
      members.emplace_back(std::move(key), parse_value(depth + 1));
      skip_whitespace();
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      expect('}');
      return Value(Value::Storage(std::move(members)));
    }
  }

  Value parse_array(int depth) {
    expect('[');
    Array items;
    skip_whitespace();
    if (peek() == ']') {
      ++pos_;
      return Value(Value::Storage(std::move(items)));
    }
    while (true) {
      items.push_back(parse_value(depth + 1));
      skip_whitespace();
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      expect(']');
      return Value(Value::Storage(std::move(items)));
    }
  }

  unsigned parse_hex4() {
    if (text_.size() - pos_ < 4) fail("truncated \\u escape");
    unsigned code = 0;
    const char* first = text_.data() + pos_;
    const auto result = std::from_chars(first, first + 4, code, 16);
    if (result.ec != std::errc() || result.ptr != first + 4) fail("invalid \\u escape");
    pos_ += 4;
    return code;
  }

  static void append_utf8(std::string& out, unsigned code) {
    if (code < 0x80) {
      out += static_cast<char>(code);
    } else if (code < 0x800) {
      out += static_cast<char>(0xC0 | (code >> 6));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
      out += static_cast<char>(0xE0 | (code >> 12));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (code >> 18));
      out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    }
  }

  // Decodes the code point after "\u", joining a surrogate pair into one code point.
  unsigned parse_unicode_escape() {
    const unsigned high = parse_hex4();
    if (high >= 0xDC00 && high <= 0xDFFF) fail("unpaired low surrogate");
    if (high < 0xD800 || high > 0xDBFF) return high;
    if (!consume_literal("\\u")) fail("unpaired high surrogate");
    const unsigned low = parse_hex4();
    if (low < 0xDC00 || low > 0xDFFF) fail("unpaired high surrogate");
    return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) fail("unterminated string");
      const char c = text_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= text_.size()) fail("unterminated string");
      switch (text_[pos_++]) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u':
          append_utf8(out, parse_unicode_escape());
          break;
        default:
          fail("invalid escape");
      }
    }
  }

  static bool is_digit(char c) { return c >= '0' && c <= '9'; }

  // Advances past one or more digits.
  void skip_digits() {
    if (!is_digit(peek())) fail("invalid number");
    while (is_digit(peek())) ++pos_;
  }

  Value parse_number() {
    const std::size_t start = pos_;
    if (peek() == '-') ++pos_;
    if (peek() == '0') {
      ++pos_;
    } else {
      skip_digits();
    }
    bool integral = true;
    if (peek() == '.') {
      integral = false;
      ++pos_;
      skip_digits();
    }
    if (peek() == 'e' || peek() == 'E') {
      integral = false;
      ++pos_;
      if (peek() == '+' || peek() == '-') ++pos_;
      skip_digits();
    }

    const char* first = text_.data() + start;
    const char* last = text_.data() + pos_;
    if (integral) {
      std::int64_t value = 0;
      const auto result = std::from_chars(first, last, value);
      if (result.ec == std::errc() && result.ptr == last) return Value(Value::Storage(value));
      // Falls through for integers beyond 64 bits, which are kept as doubles.
    }
    // from_chars ignores the process locale, unlike strtod.
    double value = 0.0;
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc() || result.ptr != last) fail("number out of range");
    return Value(Value::Storage(value));
  }

  std::string_view text_;
  std::size_t pos_ = 0;
};

template <typename T>
const T& get(const Value::Storage& storage, const char* expected) {
  if (const T* value = std::get_if<T>(&storage)) return *value;
  throw Error(std::string("json: value is not ") + expected);
}

}  // namespace

bool Value::as_bool() const { return get<bool>(storage_, "a boolean"); }

std::int64_t Value::as_int() const { return get<std::int64_t>(storage_, "an integer"); }

double Value::as_number() const {
  if (const auto* integer = std::get_if<std::int64_t>(&storage_)) {
    return static_cast<double>(*integer);
  }
  return get<double>(storage_, "a number");
}

const std::string& Value::as_string() const { return get<std::string>(storage_, "a string"); }

const Array& Value::as_array() const { return get<Array>(storage_, "an array"); }

const Object& Value::as_object() const { return get<Object>(storage_, "an object"); }

const Value* Value::find(std::string_view key) const {
  for (const auto& [name, value] : as_object()) {
    if (name == key) return &value;
  }
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  if (const Value* value = find(key)) return *value;
  throw Error("json: missing key \"" + std::string(key) + "\"");
}

Value parse(std::string_view text) { return Parser(text).parse_document(); }

}  // namespace kvb::json
