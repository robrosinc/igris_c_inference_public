#pragma once

#include "igris_c_inference/modules/policy/runtime/RlEnvConfig.h"
#include "igris_c_inference/modules/policy/utils/PolicyPipelineTypes.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace igris_c_inference::policy {

class PolicyRuntime {
public:
  explicit PolicyRuntime(std::string label);
  ~PolicyRuntime();
  PolicyRuntime(const PolicyRuntime &) = delete;
  PolicyRuntime &operator=(const PolicyRuntime &) = delete;

  int configure(const std::string &modelPath, const RlEnvConfig &config,
                std::string &error);
  void reset();
  int run(const TensorMap &inputs, TensorMap &outputs, std::string &error);
  std::vector<std::pair<std::string, std::size_t>> expectedInputSizes() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace igris_c_inference::policy
