#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace igris_c_inference::policy::observation {

using Function =
    std::function<Eigen::VectorXd(RlEnvModule &, const StepContext &)>;

class Term : public ManagerTermBase {
public:
  using ManagerTermBase::ManagerTermBase;
  virtual Eigen::VectorXd compute(const StepContext &context) = 0;
  virtual int size() const = 0;
};

struct FactoryContext {
  RlEnvModule &env;
  const std::vector<std::string> *observationJointNames{nullptr};
  const Eigen::VectorXd *observationJointReference{nullptr};
  int policyHz{0};
  std::string profileDirectory;
};

using Factory = std::function<std::unique_ptr<Term>(
    const std::string &, const YAML::Node &, const FactoryContext &)>;
using Registry = std::unordered_map<std::string, Factory>;

const Registry &registry();
std::unique_ptr<Term> makeFunctionTerm(const std::string &name,
                                       RlEnvModule &env, int size,
                                       Function function);

std::unique_ptr<Term> createCommandObservation(const std::string &name,
                                               const YAML::Node &params,
                                               const FactoryContext &context);
std::unique_ptr<Term>
createBaseAngularVelocityObservation(const std::string &name,
                                     const YAML::Node &params,
                                     const FactoryContext &context);
std::unique_ptr<Term>
createProjectedGravityObservation(const std::string &name,
                                  const YAML::Node &params,
                                  const FactoryContext &context);
std::unique_ptr<Term>
createLastActionsObservation(const std::string &name, const YAML::Node &params,
                             const FactoryContext &context);
std::unique_ptr<Term>
createRelativeJointPositionObservation(const std::string &name,
                                       const YAML::Node &params,
                                       const FactoryContext &context);
std::unique_ptr<Term>
createJointVelocityObservation(const std::string &name,
                               const YAML::Node &params,
                               const FactoryContext &context);
std::unique_ptr<Term>
createFrameAngularVelocityObservation(const std::string &name,
                                      const YAML::Node &params,
                                      const FactoryContext &context);
std::unique_ptr<Term>
createFrameProjectedGravityObservation(const std::string &name,
                                       const YAML::Node &params,
                                       const FactoryContext &context);
std::unique_ptr<Term>
createAlignedJointPositionObservation(const std::string &name,
                                      const YAML::Node &params,
                                      const FactoryContext &context);
std::unique_ptr<Term>
createAlignedJointVelocityObservation(const std::string &name,
                                      const YAML::Node &params,
                                      const FactoryContext &context);
std::unique_ptr<Term>
createAlignedLastActionsObservation(const std::string &name,
                                    const YAML::Node &params,
                                    const FactoryContext &context);

} // namespace igris_c_inference::policy::observation
