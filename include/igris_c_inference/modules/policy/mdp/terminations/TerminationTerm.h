#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace igris_c_inference::policy::termination {

class Term : public ManagerTermBase {
public:
  using ManagerTermBase::ManagerTermBase;
  virtual bool compute(const StepContext &context) = 0;
};

struct FactoryContext {
  RlEnvModule &env;
  int controlHz{0};
  int policyHz{0};
};

using Factory = std::function<std::unique_ptr<Term>(
    const std::string &, const YAML::Node &, const FactoryContext &)>;
using Registry = std::unordered_map<std::string, Factory>;

const Registry &registry();
std::unique_ptr<Term> createTimeOutTermination(const std::string &name,
                                               const YAML::Node &params,
                                               const FactoryContext &context);

} // namespace igris_c_inference::policy::termination
