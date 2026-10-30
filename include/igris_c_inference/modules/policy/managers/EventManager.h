#pragma once

#include "igris_c_inference/modules/policy/managers/ManagerBase.h"

#include <memory>
#include <string>

namespace igris_c_inference::policy {

class EventManager : public ManagerBase {
public:
  explicit EventManager(RlEnvModule &env);
  ~EventManager() override;

  int configure(const std::string &path, std::string &error);
  void reset() override;
  std::vector<TermDescriptor> termDescriptors() const override;
  int apply(const std::string &mode, const LifecycleContext &context,
            std::string &error);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace igris_c_inference::policy
