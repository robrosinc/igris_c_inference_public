#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"
#include "igris_c_inference/modules/policy/utils/PolicyPipelineTypes.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace igris_c_inference::policy::command {

using Clock = std::chrono::steady_clock;

struct PrepareContext {
  const PolicyControlContext *committed{nullptr};
  const SignalMap *preparedSignals{nullptr};
  uint64_t policyStep{0};
};

struct StepFeedback {
  const TensorMap *outputs{nullptr};
  const ActionFrame *actions{nullptr};
  uint64_t policyStep{0};
};

class Term : public ManagerTermBase {
public:
  using ManagerTermBase::ManagerTermBase;
  virtual bool start() { return true; }
  virtual void stop() {}
  virtual void resetHeadingAlignment() {}
  virtual bool waitUntilReady(std::chrono::milliseconds) { return true; }
  virtual SignalMap prepare(const StepContext &step,
                            const PrepareContext &context) = 0;
  virtual void commit(const StepContext &, const StepFeedback &) {}
  virtual SignalMap resetSignals() const = 0;
  virtual std::vector<SignalDescriptor> signalDescriptors() const = 0;
};

struct FactoryContext {
  RlEnvModule &env;
  int policyHz{0};
  std::string profileDirectory;
  std::string modelPath;
  std::unordered_map<std::string, std::string> resourcePaths;
};

using Factory = std::function<std::unique_ptr<Term>(
    const std::string &, const YAML::Node &, const FactoryContext &)>;
using Registry = std::unordered_map<std::string, Factory>;

const Registry &registry();

std::unique_ptr<Term> createVelocityCommand(const std::string &name,
                                            const YAML::Node &config,
                                            const FactoryContext &context);
std::unique_ptr<Term>
createSonicMotionReferenceCommand(const std::string &name,
                                  const YAML::Node &config,
                                  const FactoryContext &context);
std::unique_ptr<Term>
createMotionReferenceCommand(const std::string &name, const YAML::Node &config,
                             const FactoryContext &context);

} // namespace igris_c_inference::policy::command
