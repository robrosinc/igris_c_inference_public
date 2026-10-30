#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace igris_c_inference::policy::event {

class Term : public ManagerTermBase {
public:
  using ManagerTermBase::ManagerTermBase;
  virtual void apply(const LifecycleContext &context) = 0;
};

struct FactoryContext {
  RlEnvModule &env;
};

using Factory = std::function<std::unique_ptr<Term>(
    const std::string &, const YAML::Node &, const FactoryContext &)>;
using Registry = std::unordered_map<std::string, Factory>;

const Registry &registry();

} // namespace igris_c_inference::policy::event
