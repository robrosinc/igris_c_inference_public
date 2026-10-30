#include "igris_c_inference/modules/policy/managers/CommandManager.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/mdp/commands/CommandTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace igris_c_inference::policy {
namespace {

void mergeSignals(SignalMap &target, SignalMap source) {
  for (auto &entry : source) {
    if (entry.first.empty())
      throw std::runtime_error("produced a signal with an empty name");
    if (entry.second.value.size() == 0)
      throw std::runtime_error("signal '" + entry.first + "' is empty");
    for (Eigen::Index index = 0; index < entry.second.value.size(); ++index) {
      if (!std::isfinite(entry.second.value(index)))
        throw std::runtime_error("signal '" + entry.first +
                                 "' is non-finite at index " +
                                 std::to_string(index));
    }
    if (!target.emplace(entry.first, std::move(entry.second)).second)
      throw std::runtime_error("produced duplicate signal '" + entry.first +
                               "'");
  }
}

void validateSignals(const std::string &termName,
                     const std::vector<SignalDescriptor> &descriptors,
                     const SignalMap &signals) {
  for (const auto &[name, signal] : signals) {
    const auto descriptor =
        std::find_if(descriptors.begin(), descriptors.end(),
                     [&name](const SignalDescriptor &candidate) {
                       return candidate.name == name;
                     });
    if (descriptor == descriptors.end())
      throw std::runtime_error("command term '" + termName +
                               "' produced undeclared signal '" + name + "'");
    if (signal.value.size() != descriptor->size)
      throw std::runtime_error(
          "command term '" + termName + "' produced signal '" + name +
          "' with size " + std::to_string(signal.value.size()) + ", expected " +
          std::to_string(descriptor->size));
  }
}

} // namespace

class CommandManager::Impl {
public:
  struct Entry {
    std::string name;
    std::string function;
    std::vector<std::string> dependencies;
    std::vector<SignalDescriptor> signals;
    std::unique_ptr<command::Term> term;
  };

  std::vector<Entry> entries;
  std::vector<std::size_t> executionOrder;
  std::unordered_map<std::string, int> signalSizes;
  PolicyControlContext committed;
  PolicyControlContext pending;
  bool prepared{false};
  bool termsNeedReset{false};
  bool initialized{false};
  uint64_t generation{0};
  uint64_t pendingGeneration{0};
  mutable std::mutex contextMutex;
  mutable std::mutex termExecutionMutex;
};

CommandManager::CommandManager(RlEnvModule &env)
    : ManagerBase(env), impl_(std::make_unique<Impl>()) {}
CommandManager::~CommandManager() { stop(); }

int CommandManager::configure(
    const std::string &path, int policyHz, const std::string &modelPath,
    const std::unordered_map<std::string, std::string> &resourcePaths,
    std::string &error) {
  stop();
  try {
    if (policyHz <= 0)
      throw std::runtime_error("policy_hz must be positive");
    const YAML::Node root = YAML::LoadFile(path);
    yaml::checkKeys(root, {"igris_c_inference_commands"}, "command root");
    const YAML::Node config = root["igris_c_inference_commands"];
    yaml::checkKeys(config, {"terms"}, "commands");
    const YAML::Node terms = config["terms"];
    if (!terms || !terms.IsMap())
      throw std::runtime_error("commands.terms must be a map");

    impl_->entries.clear();
    impl_->executionOrder.clear();
    impl_->signalSizes.clear();
    const auto &factories = command::registry();
    const command::FactoryContext context{
        env(), policyHz, std::filesystem::path(path).parent_path().string(),
        modelPath, resourcePaths};
    std::unordered_set<std::string> names;
    for (const auto &entry : terms) {
      const std::string name = entry.first.as<std::string>();
      const YAML::Node node = entry.second;
      if (name.empty() || !names.insert(name).second)
        throw std::runtime_error(
            "command term names must be non-empty and unique");
      const std::string where = "commands.terms." + name;
      yaml::checkKeys(node, {"function", "source", "depends_on", "params"},
                      where);
      const std::string function =
          yaml::required<std::string>(node, "function", where);
      const auto factory = factories.find(function);
      if (factory == factories.end())
        throw std::runtime_error("unknown command function: " + function);

      Impl::Entry parsed;
      parsed.name = name;
      parsed.function = function;
      if (node["depends_on"])
        parsed.dependencies =
            yaml::readNames(node["depends_on"], where + ".depends_on", true);
      parsed.term = factory->second(name, node, context);
      if (!parsed.term)
        throw std::runtime_error("command factory returned null: " + function);
      parsed.signals = parsed.term->signalDescriptors();
      if (parsed.signals.empty())
        throw std::runtime_error("command term '" + name +
                                 "' declares no signals");
      for (const SignalDescriptor &signal : parsed.signals) {
        if (signal.name.empty() || signal.size <= 0)
          throw std::runtime_error("command term '" + name +
                                   "' declared an invalid signal");
        if (!impl_->signalSizes.emplace(signal.name, signal.size).second)
          throw std::runtime_error("command terms declare duplicate signal '" +
                                   signal.name + "'");
      }
      impl_->entries.push_back(std::move(parsed));
    }
    for (const auto &entry : impl_->entries) {
      for (const std::string &dependency : entry.dependencies) {
        if (names.count(dependency) == 0U)
          throw std::runtime_error("command dependency is not defined: " +
                                   dependency);
      }
    }

    std::unordered_set<std::string> completed;
    while (impl_->executionOrder.size() < impl_->entries.size()) {
      bool progressed = false;
      for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
        const auto &entry = impl_->entries[index];
        if (completed.count(entry.name) != 0U)
          continue;
        if (std::any_of(entry.dependencies.begin(), entry.dependencies.end(),
                        [&completed](const std::string &name) {
                          return completed.count(name) == 0U;
                        }))
          continue;
        impl_->executionOrder.push_back(index);
        completed.insert(entry.name);
        progressed = true;
      }
      if (!progressed)
        throw std::runtime_error("command dependencies contain a cycle");
    }
    reset();
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    stop();
    return -1;
  }
}

bool CommandManager::initialize(std::string &error) {
  try {
    for (auto &entry : impl_->entries) {
      if (!entry.term->start())
        throw std::runtime_error("failed to start command '" + entry.name +
                                 "'");
    }
    impl_->initialized = true;
    error.clear();
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    stop();
    return false;
  }
}

void CommandManager::stop() {
  std::lock_guard<std::mutex> executionLock(impl_->termExecutionMutex);
  for (auto &entry : impl_->entries)
    entry.term->stop();
  impl_->initialized = false;
}

bool CommandManager::waitUntilReady(std::chrono::milliseconds timeout,
                                    std::string &error) {
  if (!impl_->initialized) {
    error = "command manager is not initialized";
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (auto &entry : impl_->entries) {
    const auto now = std::chrono::steady_clock::now();
    if (!entry.term->waitUntilReady(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::max(deadline - now,
                         std::chrono::steady_clock::duration::zero())))) {
      error = "timed out waiting for command '" + entry.name + "'";
      return false;
    }
  }
  error.clear();
  return true;
}

void CommandManager::reset() {
  std::lock_guard<std::mutex> executionLock(impl_->termExecutionMutex);
  SignalMap initialSignals;
  for (auto &entry : impl_->entries) {
    entry.term->reset();
    SignalMap signals = entry.term->resetSignals();
    validateSignals(entry.name, entry.signals, signals);
    mergeSignals(initialSignals, std::move(signals));
  }
  std::lock_guard<std::mutex> lock(impl_->contextMutex);
  ++impl_->generation;
  impl_->committed = PolicyControlContext{};
  impl_->committed.signals = std::move(initialSignals);
  impl_->pending = impl_->committed;
  impl_->prepared = false;
  impl_->termsNeedReset = false;
  impl_->pendingGeneration = 0;
}

void CommandManager::resetHeadingAlignment() {
  std::lock_guard<std::mutex> executionLock(impl_->termExecutionMutex);
  for (auto &entry : impl_->entries)
    entry.term->resetHeadingAlignment();
}

std::vector<TermDescriptor> CommandManager::termDescriptors() const {
  std::vector<TermDescriptor> result;
  result.reserve(impl_->entries.size());
  for (const auto &entry : impl_->entries) {
    result.push_back(TermDescriptor{entry.name,
                                    entry.function,
                                    {},
                                    "policy",
                                    0,
                                    entry.dependencies,
                                    entry.signals});
  }
  return result;
}

bool CommandManager::getSignal(const std::string &name,
                               const StepContext &context,
                               Eigen::VectorXd &value) const {
  PolicySignal signal;
  if (!getSignal(name, context, signal))
    return false;
  value = std::move(signal.value);
  return true;
}

bool CommandManager::getSignal(const std::string &name,
                               const StepContext &context,
                               PolicySignal &signal) const {
  std::lock_guard<std::mutex> lock(impl_->contextMutex);
  const PolicyControlContext &signals =
      context.phase == ExecutionPhase::POLICY && impl_->prepared
          ? impl_->pending
          : impl_->committed;
  const auto found = signals.signals.find(name);
  if (found == signals.signals.end())
    return false;
  signal = found->second;
  return true;
}

int CommandManager::signalSize(const std::string &name) const {
  const auto found = impl_->signalSizes.find(name);
  return found == impl_->signalSizes.end() ? 0 : found->second;
}

uint64_t CommandManager::policyStep() const {
  std::lock_guard<std::mutex> lock(impl_->contextMutex);
  return impl_->committed.policyStep;
}

int CommandManager::prepare(const StepContext &step, std::string &error) {
  try {
    std::lock_guard<std::mutex> executionLock(impl_->termExecutionMutex);
    bool resetTerms = false;
    uint64_t generation = 0;
    PolicyControlContext committed;
    {
      std::lock_guard<std::mutex> lock(impl_->contextMutex);
      resetTerms = impl_->termsNeedReset;
      impl_->termsNeedReset = false;
      generation = impl_->generation;
      committed = impl_->committed;
    }
    if (resetTerms) {
      for (auto &entry : impl_->entries)
        entry.term->reset();
    }

    PolicyControlContext next = committed;
    next.signals.clear();
    for (const std::size_t index : impl_->executionOrder) {
      auto &entry = impl_->entries[index];
      const command::PrepareContext context{&committed, &next.signals,
                                            committed.policyStep};
      try {
        SignalMap signals = entry.term->prepare(step, context);
        validateSignals(entry.name, entry.signals, signals);
        mergeSignals(next.signals, std::move(signals));
      } catch (const std::exception &exception) {
        throw std::runtime_error("command term '" + entry.name +
                                 "' failed: " + exception.what());
      }
    }
    {
      std::lock_guard<std::mutex> lock(impl_->contextMutex);
      if (generation != impl_->generation) {
        error.clear();
        return 0;
      }
      impl_->pending = std::move(next);
      impl_->prepared = true;
      impl_->pendingGeneration = generation;
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    std::lock_guard<std::mutex> lock(impl_->contextMutex);
    impl_->prepared = false;
    return -1;
  }
}

void CommandManager::commit(const StepContext &context,
                            const TensorMap &outputs,
                            const ActionFrame &actions) {
  std::lock_guard<std::mutex> executionLock(impl_->termExecutionMutex);
  PolicyControlContext pending;
  uint64_t step = 0;
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(impl_->contextMutex);
    if (!impl_->prepared || impl_->pendingGeneration != impl_->generation)
      return;
    pending = impl_->pending;
    step = impl_->committed.policyStep;
    generation = impl_->generation;
  }
  const command::StepFeedback feedback{&outputs, &actions, step};
  for (const std::size_t index : impl_->executionOrder) {
    auto &entry = impl_->entries[index];
    try {
      entry.term->commit(context, feedback);
    } catch (const std::exception &exception) {
      throw std::runtime_error("command term '" + entry.name +
                               "' commit failed: " + exception.what());
    }
  }
  std::lock_guard<std::mutex> lock(impl_->contextMutex);
  if (generation != impl_->generation)
    return;
  pending.policyStep = step + 1U;
  impl_->committed = std::move(pending);
  impl_->prepared = false;
}

} // namespace igris_c_inference::policy
