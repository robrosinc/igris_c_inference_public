#pragma once

#include "igris_c_inference/core/types.hpp"
#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace igris_c_inference::policy::action {

class Term : public ManagerTermBase {
public:
  using ManagerTermBase::ManagerTermBase;

  virtual Eigen::VectorXd process(const Eigen::VectorXd &actions,
                                  const StepContext &context);
  virtual Eigen::VectorXd resetValue(int size) const;
  virtual void apply(const Eigen::VectorXd &actions, const StepContext &context,
                     InferenceCommand &command) const = 0;
  virtual int actionSize() const = 0;
  virtual bool consumesPolicyOutput() const { return true; }
};

struct FactoryContext {
  RlEnvModule &env;
};

using Factory = std::function<std::unique_ptr<Term>(
    const std::string &, const YAML::Node &, const FactoryContext &)>;
using Registry = std::unordered_map<std::string, Factory>;

const Registry &registry();

std::unique_ptr<Term> createJointPositionAction(const std::string &name,
                                                const YAML::Node &config,
                                                const FactoryContext &context);
std::unique_ptr<Term>
createSafeJointPositionAction(const std::string &name, const YAML::Node &config,
                              const FactoryContext &context);
std::unique_ptr<Term> createMotionCommandAction(const std::string &name,
                                                const YAML::Node &config,
                                                const FactoryContext &context);

} // namespace igris_c_inference::policy::action
