#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace igris_c_inference::policy {

struct OutputBinding {
  std::string name;
  std::string tensorName;
  std::size_t offset = 0;
  std::size_t size = 0;
};

struct RlEnvConfig {
  int rateHz = 50;
  std::string commandPath;
  std::string observationPath;
  std::string actionPath;
  std::string terminationPath;
  std::string eventPath;
  std::vector<std::string> inputNames;
  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<OutputBinding> outputs;
};

int loadRlEnvConfig(const std::string &path, RlEnvConfig &config,
                    std::string &error);

} // namespace igris_c_inference::policy
