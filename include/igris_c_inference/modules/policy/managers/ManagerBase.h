#pragma once

#include "igris_c_inference/modules/policy/utils/PolicyPipelineTypes.h"

#include <string>
#include <utility>
#include <vector>

namespace igris_c_inference {
class RlEnvModule;
}

namespace igris_c_inference::policy {

class ManagerBase {
public:
  explicit ManagerBase(RlEnvModule &env) : env_(env) {}
  virtual ~ManagerBase() = default;

  ManagerBase(const ManagerBase &) = delete;
  ManagerBase &operator=(const ManagerBase &) = delete;

  virtual void reset() = 0;
  virtual std::vector<TermDescriptor> termDescriptors() const = 0;

protected:
  RlEnvModule &env() { return env_; }
  const RlEnvModule &env() const { return env_; }

private:
  RlEnvModule &env_;
};

class ManagerTermBase {
public:
  ManagerTermBase(std::string name, RlEnvModule &env)
      : name_(std::move(name)), env_(env) {}
  virtual ~ManagerTermBase() = default;

  ManagerTermBase(const ManagerTermBase &) = delete;
  ManagerTermBase &operator=(const ManagerTermBase &) = delete;

  virtual void reset() {}
  const std::string &name() const { return name_; }

protected:
  RlEnvModule &env() { return env_; }
  const RlEnvModule &env() const { return env_; }

private:
  std::string name_;
  RlEnvModule &env_;
};

} // namespace igris_c_inference::policy
