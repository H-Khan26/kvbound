#pragma once

#include <cstdlib>
#include <map>
#include <string>
#include <utility>

#include "kvbound/error.h"

// Command-line options of the kvb-* tools: --name value pairs.
class Args {
 public:
  Args(int argc, char** argv) {
    if (argc % 2 == 0) throw kvb::Error("every option needs a value");
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string key = argv[i];
      if (key.rfind("--", 0) != 0) throw kvb::Error("expected --option, got " + key);
      values_[key.substr(2)] = argv[i + 1];
    }
  }
  std::string required(const std::string& key) {
    const auto it = values_.find(key);
    if (it == values_.end()) throw kvb::Error("missing --" + key);
    return take(it);
  }
  std::string str(const std::string& key, const std::string& fallback) {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : take(it);
  }
  int integer(const std::string& key, int fallback) {
    return std::atoi(str(key, std::to_string(fallback)).c_str());
  }
  void expect_consumed() const {
    if (!values_.empty()) throw kvb::Error("unknown option --" + values_.begin()->first);
  }

 private:
  std::string take(std::map<std::string, std::string>::iterator it) {
    std::string value = std::move(it->second);
    values_.erase(it);
    return value;
  }
  std::map<std::string, std::string> values_;
};
