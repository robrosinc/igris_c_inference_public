#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <cstdint>
#include <memory>
#include <string>

namespace igris_c_inference::policy {

enum class TerminationUpdatePhase { CONTROL, POLICY };
enum class TerminationOutcomeType { NONE, STOP };

struct TerminationRequest {
  TerminationOutcomeType outcome{TerminationOutcomeType::NONE};
  std::string termName;
  uint64_t sequence{0};
};

class TerminationManager : public ManagerBase {
public:
  explicit TerminationManager(RlEnvModule &env);
  ~TerminationManager() override;

  int configure(const std::string &path, int controlHz, int policyHz,
                std::string &error);
  void reset() override;
  std::vector<TermDescriptor> termDescriptors() const override;
  int update(const StepContext &context, TerminationUpdatePhase phase,
             std::string &error);
  bool consumeRequest(TerminationRequest &request);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace igris_c_inference::policy
