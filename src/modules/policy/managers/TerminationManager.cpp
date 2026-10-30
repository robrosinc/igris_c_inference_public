#include "igris_c_inference/modules/policy/managers/TerminationManager.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/mdp/terminations/TerminationTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <yaml-cpp/yaml.h>

#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace igris_c_inference::policy {

class TerminationManager::Impl {
public:
  struct Entry {
    std::string name;
    std::string function;
    TerminationUpdatePhase phase{TerminationUpdatePhase::POLICY};
    TerminationOutcomeType outcome{TerminationOutcomeType::NONE};
    std::unique_ptr<termination::Term> term;
    bool active{false};
  };

  std::vector<Entry> entries;
  TerminationRequest pending;
  uint64_t sequence{0};
  bool hasPending{false};
  mutable std::mutex stateMutex;
  mutable std::mutex requestMutex;
};

TerminationManager::TerminationManager(RlEnvModule &env)
    : ManagerBase(env), impl_(std::make_unique<Impl>()) {}
TerminationManager::~TerminationManager() = default;

int TerminationManager::configure(const std::string &path, int controlHz,
                                  int policyHz, std::string &error) {
  try {
    if (controlHz <= 0 || policyHz <= 0)
      throw std::runtime_error(
          "termination control_hz and policy_hz must be positive");
    const YAML::Node root = YAML::LoadFile(path);
    yaml::checkKeys(root, {"igris_c_inference_terminations"},
                    "termination root");
    const YAML::Node config = root["igris_c_inference_terminations"];
    yaml::checkKeys(config, {"terms"}, "terminations");
    const YAML::Node terms = config["terms"];
    if (!terms || !terms.IsMap())
      throw std::runtime_error("terminations.terms must be a map");

    impl_->entries.clear();
    const auto &factories = termination::registry();
    const termination::FactoryContext context{env(), controlHz, policyHz};
    std::unordered_set<std::string> names;
    for (const auto &entry : terms) {
      const std::string name = entry.first.as<std::string>();
      const YAML::Node node = entry.second;
      if (name.empty() || !names.insert(name).second)
        throw std::runtime_error(
            "termination term names must be non-empty and unique");
      const std::string where = "terminations.terms." + name;
      yaml::checkKeys(node, {"function", "update_at", "params", "outcome"},
                      where);
      const std::string function =
          yaml::required<std::string>(node, "function", where);
      const auto factory = factories.find(function);
      if (factory == factories.end())
        throw std::runtime_error("unknown termination function: " + function);

      Impl::Entry parsed;
      parsed.name = name;
      parsed.function = function;
      const std::string updateAt =
          node["update_at"] ? node["update_at"].as<std::string>() : "policy";
      if (updateAt == "control")
        parsed.phase = TerminationUpdatePhase::CONTROL;
      else if (updateAt == "policy")
        parsed.phase = TerminationUpdatePhase::POLICY;
      else
        throw std::runtime_error(where +
                                 ".update_at must be control or policy");

      const YAML::Node outcome = yaml::optionalMap(node, "outcome", where);
      yaml::checkKeys(outcome, {"type"}, where + ".outcome");
      const std::string outcomeType =
          yaml::required<std::string>(outcome, "type", where + ".outcome");
      if (outcomeType != "stop")
        throw std::runtime_error(where + ".outcome.type must be stop");
      parsed.outcome = TerminationOutcomeType::STOP;
      parsed.term = factory->second(
          name, yaml::optionalMap(node, "params", where), context);
      if (!parsed.term)
        throw std::runtime_error("termination factory returned null: " +
                                 function);
      impl_->entries.push_back(std::move(parsed));
    }
    reset();
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

void TerminationManager::reset() {
  {
    std::lock_guard<std::mutex> stateLock(impl_->stateMutex);
    for (auto &entry : impl_->entries) {
      entry.term->reset();
      entry.active = false;
    }
  }
  std::lock_guard<std::mutex> requestLock(impl_->requestMutex);
  impl_->pending = TerminationRequest{};
  impl_->hasPending = false;
}

std::vector<TermDescriptor> TerminationManager::termDescriptors() const {
  std::vector<TermDescriptor> result;
  result.reserve(impl_->entries.size());
  for (const auto &entry : impl_->entries) {
    result.push_back(TermDescriptor{
        entry.name,
        entry.function,
        "stop",
        entry.phase == TerminationUpdatePhase::CONTROL ? "control" : "policy",
        1,
        {},
        {}});
  }
  return result;
}

int TerminationManager::update(const StepContext &context,
                               TerminationUpdatePhase phase,
                               std::string &error) {
  try {
    std::lock_guard<std::mutex> stateLock(impl_->stateMutex);
    for (auto &entry : impl_->entries) {
      if (entry.phase != phase)
        continue;
      const bool active = entry.term->compute(context);
      const bool rising = active && !entry.active;
      entry.active = active;
      if (!rising)
        continue;
      std::lock_guard<std::mutex> requestLock(impl_->requestMutex);
      if (!impl_->hasPending) {
        impl_->pending.outcome = entry.outcome;
        impl_->pending.termName = entry.name;
        impl_->pending.sequence = ++impl_->sequence;
        impl_->hasPending = true;
      }
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

bool TerminationManager::consumeRequest(TerminationRequest &request) {
  std::lock_guard<std::mutex> requestLock(impl_->requestMutex);
  if (!impl_->hasPending)
    return false;
  request = impl_->pending;
  impl_->hasPending = false;
  return true;
}

} // namespace igris_c_inference::policy
