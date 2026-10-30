#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"
#include "igris_c_inference/modules/policy/utils/PolicyPipelineTypes.h"

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

namespace igris_c_inference::policy {

class CommandManager : public ManagerBase {
public:
  explicit CommandManager(RlEnvModule &env);
  ~CommandManager() override;

  int configure(
      const std::string &path, int policyHz, const std::string &modelPath,
      const std::unordered_map<std::string, std::string> &resourcePaths,
      std::string &error);
  bool initialize(std::string &error);
  void stop();
  bool waitUntilReady(std::chrono::milliseconds timeout, std::string &error);
  void reset() override;
  void resetHeadingAlignment();
  std::vector<TermDescriptor> termDescriptors() const override;
  bool getSignal(const std::string &name, const StepContext &context,
                 Eigen::VectorXd &value) const;
  bool getSignal(const std::string &name, const StepContext &context,
                 PolicySignal &signal) const;
  int signalSize(const std::string &name) const;
  uint64_t policyStep() const;
  int prepare(const StepContext &context, std::string &error);
  void commit(const StepContext &context, const TensorMap &outputs,
              const ActionFrame &actions);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace igris_c_inference::policy
