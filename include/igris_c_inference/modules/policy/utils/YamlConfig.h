#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace igris_c_inference::policy::yaml {

template <typename T>
T required(const YAML::Node &node, const char *key, const std::string &where) {
  if (!node || !node[key])
    throw std::runtime_error(where + ": missing '" + key + "'");
  return node[key].as<T>();
}

inline void checkKeys(const YAML::Node &node,
                      std::initializer_list<std::string_view> allowed,
                      const std::string &where) {
  if (!node || !node.IsMap())
    throw std::runtime_error(where + " must be a map");
  std::unordered_set<std::string> keys;
  for (const std::string_view key : allowed)
    keys.emplace(key);
  for (const auto &entry : node) {
    const std::string key = entry.first.as<std::string>();
    if (keys.count(key) == 0U)
      throw std::runtime_error(where + ": unknown key '" + key + "'");
  }
}

inline YAML::Node optionalMap(const YAML::Node &node, const char *key,
                              const std::string &where) {
  if (!node || !node[key])
    return YAML::Node(YAML::NodeType::Map);
  if (!node[key].IsMap())
    throw std::runtime_error(where + "." + key + " must be a map");
  return node[key];
}

inline std::vector<std::string> readNames(const YAML::Node &node,
                                          const std::string &where,
                                          bool allow_empty = false) {
  if (!node || !node.IsSequence() || (!allow_empty && node.size() == 0U))
    throw std::runtime_error(where + (allow_empty
                                          ? " must be a sequence"
                                          : " must be a non-empty sequence"));
  std::vector<std::string> values;
  std::unordered_set<std::string> seen;
  for (const auto &entry : node) {
    const std::string value = entry.as<std::string>();
    if (value.empty() || !seen.insert(value).second)
      throw std::runtime_error(where + " contains an empty or duplicate name");
    values.push_back(value);
  }
  return values;
}

inline Eigen::VectorXd readVector(const YAML::Node &node, std::size_t size,
                                  const std::string &where) {
  if (!node || !node.IsSequence() || node.size() != size)
    throw std::runtime_error(where + " has an unexpected vector size");
  Eigen::VectorXd values(static_cast<Eigen::Index>(size));
  for (std::size_t index = 0; index < size; ++index)
    values(static_cast<Eigen::Index>(index)) = node[index].as<double>();
  if (!values.allFinite())
    throw std::runtime_error(where + " must contain finite values");
  return values;
}

} // namespace igris_c_inference::policy::yaml
