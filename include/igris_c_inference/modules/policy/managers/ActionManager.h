#pragma once

#include "igris_c_inference/core/types.hpp"
#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <Eigen/Dense>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace igris_c_inference::policy {

class ActionManager : public ManagerBase {
public:
  explicit ActionManager(RlEnvModule &env);
  ~ActionManager() override;
  ActionManager(const ActionManager &) = delete;
  ActionManager &operator=(const ActionManager &) = delete;

  int configure(const std::string &path, double policyHz, std::string &error);
  void reset() override;
  std::vector<TermDescriptor> termDescriptors() const override;
  int process(const TensorMap &outputs, const StepContext &context,
              ActionFrame &processedActions, std::string &error);
  int build(const ActionFrame &actions, const StepContext &context,
            InferenceCommand &command, std::string &error) const;

  int termSize(const std::string &termName) const;
  std::vector<std::pair<std::string, int>> outputSizes() const;
  bool getRawActions(const std::string &termName, Eigen::VectorXd &actions,
                     uint64_t *sequence = nullptr) const;
  bool getProcessedActions(const std::string &termName,
                           Eigen::VectorXd &actions,
                           uint64_t *sequence = nullptr) const;
  bool getJointMetadata(const std::string &termName,
                        std::vector<std::string> &jointNames,
                        Eigen::VectorXd &actionScale) const;
  ActionFrame actionFrame() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace igris_c_inference::policy
