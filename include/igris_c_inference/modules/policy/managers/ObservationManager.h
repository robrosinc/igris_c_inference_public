#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace igris_c_inference::policy {

enum class ObservationUpdatePhase { CONTROL, POLICY };

class ObservationManager : public ManagerBase {
public:
  explicit ObservationManager(RlEnvModule &env);
  ~ObservationManager() override;

  int configure(const std::string &path, int controlHz, int policyHz,
                std::string &error);
  void reset() override;
  std::vector<TermDescriptor> termDescriptors() const override;
  int update(const StepContext &context, ObservationUpdatePhase phase,
             std::string &error);
  std::vector<std::pair<std::string, int>> groupSizes() const;
  int buildLatest(TensorMap &groups, std::string &error);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace igris_c_inference::policy
