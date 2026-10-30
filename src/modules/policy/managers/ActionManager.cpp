#include "igris_c_inference/modules/policy/managers/ActionManager.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/mdp/actions/ActionTerm.h"
#include "igris_c_inference/modules/policy/mdp/actions/JointActionTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace igris_c_inference::policy {
namespace {

void requireFinite(const Eigen::VectorXd &value, const std::string &where) {
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    if (!std::isfinite(value(index)))
      throw std::runtime_error(where + " is non-finite at index " +
                               std::to_string(index));
  }
}

} // namespace

class ActionManager::Impl {
public:
  struct Entry {
    std::string name;
    std::string function;
    std::string output;
    std::unique_ptr<action::Term> term;
    Eigen::VectorXd raw;
    Eigen::VectorXd filtered;
    Eigen::VectorXd processed;
    double clipMin = -std::numeric_limits<double>::infinity();
    double clipMax = std::numeric_limits<double>::infinity();
    double lpfCutoffHz = -1.0;
    bool filterInitialized = false;

    int size() const { return term->actionSize(); }
  };

  std::vector<Entry> entries;
  std::vector<std::size_t> applyOrder;
  std::unordered_map<std::string, std::size_t> entryIndices;
  ActionFrame lastActions;
  uint64_t processSequence = 0;
  double policyDt = 0.0;
  mutable std::mutex mutex;
};

ActionManager::ActionManager(RlEnvModule &env)
    : ManagerBase(env), impl_(std::make_unique<Impl>()) {}
ActionManager::~ActionManager() = default;

int ActionManager::configure(const std::string &path, double policyHz,
                             std::string &error) {
  try {
    if (!std::isfinite(policyHz) || policyHz <= 0.0)
      throw std::runtime_error("policy_hz must be positive");
    const YAML::Node root = YAML::LoadFile(path);
    yaml::checkKeys(root, {"igris_c_inference_actions"}, "action root");
    const YAML::Node config = root["igris_c_inference_actions"];
    yaml::checkKeys(config, {"terms", "apply_order"}, "actions");
    const YAML::Node terms = config["terms"];
    if (!terms || !terms.IsMap() || terms.size() == 0U)
      throw std::runtime_error("actions.terms must be a non-empty map");

    impl_->entries.clear();
    impl_->applyOrder.clear();
    impl_->entryIndices.clear();
    impl_->policyDt = 1.0 / policyHz;
    const auto &factories = action::registry();
    const action::FactoryContext context{env()};
    std::unordered_set<std::string> names;
    std::unordered_set<std::string> outputs;
    for (const auto &termEntry : terms) {
      Impl::Entry entry;
      entry.name = termEntry.first.as<std::string>();
      const YAML::Node termNode = termEntry.second;
      if (entry.name.empty() || !names.insert(entry.name).second)
        throw std::runtime_error(
            "action term names must be non-empty and unique");
      const std::string where = "actions.terms." + entry.name;
      entry.function = yaml::required<std::string>(termNode, "function", where);
      const auto factory = factories.find(entry.function);
      if (factory == factories.end())
        throw std::runtime_error("unknown action function: " + entry.function);
      entry.term = factory->second(entry.name, termNode, context);
      if (!entry.term)
        throw std::runtime_error("action factory returned null: " +
                                 entry.function);
      if (entry.term->consumesPolicyOutput()) {
        entry.output = yaml::required<std::string>(termNode, "output", where);
        if (entry.output.empty() || !outputs.insert(entry.output).second)
          throw std::runtime_error(
              "action outputs must be non-empty and unique");
        if (entry.term->actionSize() <= 0)
          throw std::runtime_error(where + " has no positive action size");
      } else {
        if (termNode["output"])
          throw std::runtime_error(where + " does not consume a policy output");
        if (entry.term->actionSize() != 0)
          throw std::runtime_error(
              where +
              " must have zero action size when no policy output is consumed");
      }
      if (termNode["output_clip"]) {
        const YAML::Node clip = termNode["output_clip"];
        yaml::checkKeys(clip, {"min", "max"}, where + ".output_clip");
        entry.clipMin = clip["min"].as<double>();
        entry.clipMax = clip["max"].as<double>();
        if (!std::isfinite(entry.clipMin) || !std::isfinite(entry.clipMax) ||
            entry.clipMin >= entry.clipMax)
          throw std::runtime_error(where + ".output_clip is invalid");
      }
      entry.lpfCutoffHz = termNode["lpf_cutoff_hz"]
                              ? termNode["lpf_cutoff_hz"].as<double>()
                              : -1.0;
      if (!std::isfinite(entry.lpfCutoffHz) ||
          (entry.lpfCutoffHz <= 0.0 && entry.lpfCutoffHz != -1.0))
        throw std::runtime_error(where +
                                 ".lpf_cutoff_hz must be -1 or positive");
      impl_->entryIndices.emplace(entry.name, impl_->entries.size());
      impl_->entries.push_back(std::move(entry));
    }

    std::unordered_set<std::string> orderedNames;
    if (config["apply_order"]) {
      const YAML::Node applyOrder = config["apply_order"];
      if (!applyOrder.IsSequence())
        throw std::runtime_error("actions.apply_order must be a sequence");
      for (const auto &nameNode : applyOrder) {
        if (!nameNode.IsScalar())
          throw std::runtime_error(
              "actions.apply_order entries must be term names");
        const std::string name = nameNode.as<std::string>();
        const auto found = impl_->entryIndices.find(name);
        if (found == impl_->entryIndices.end())
          throw std::runtime_error(
              "actions.apply_order references unknown term: " + name);
        if (!orderedNames.insert(name).second)
          throw std::runtime_error(
              "actions.apply_order contains duplicate term: " + name);
        impl_->applyOrder.push_back(found->second);
      }
    }
    for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
      if (orderedNames.count(impl_->entries[index].name) == 0U)
        impl_->applyOrder.push_back(index);
    }
    reset();
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

void ActionManager::reset() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (auto &entry : impl_->entries) {
    entry.term->reset();
    entry.raw = entry.term->resetValue(entry.size());
    entry.filtered.resize(0);
    entry.processed = entry.term->resetValue(entry.size());
    if (entry.raw.size() != entry.size() || !entry.raw.allFinite() ||
        entry.processed.size() != entry.size() || !entry.processed.allFinite())
      throw std::runtime_error("action term '" + entry.name +
                               "' returned an invalid reset value");
    entry.filterInitialized = false;
  }
  impl_->lastActions.clear();
  for (const auto &entry : impl_->entries) {
    if (entry.term->consumesPolicyOutput())
      impl_->lastActions.emplace(entry.name, entry.processed);
  }
  impl_->processSequence = 0;
}

std::vector<TermDescriptor> ActionManager::termDescriptors() const {
  std::vector<TermDescriptor> result;
  result.reserve(impl_->entries.size());
  for (const auto &entry : impl_->entries) {
    result.push_back(TermDescriptor{entry.name,
                                    entry.function,
                                    entry.output,
                                    "policy",
                                    entry.size(),
                                    {},
                                    {}});
  }
  return result;
}

int ActionManager::process(const TensorMap &outputs, const StepContext &context,
                           ActionFrame &processedActions, std::string &error) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto &entry : impl_->entries) {
      if (!entry.term->consumesPolicyOutput())
        continue;
      const auto output = outputs.find(entry.output);
      if (output == outputs.end())
        throw std::runtime_error("policy action output is missing: " +
                                 entry.output);
      if (output->second.size() != entry.size())
        throw std::runtime_error("action term '" + entry.name + "' expected " +
                                 std::to_string(entry.size()) +
                                 " values, got " +
                                 std::to_string(output->second.size()));
      requireFinite(output->second, "action term '" + entry.name + "'");
      entry.raw = output->second;
      const Eigen::VectorXd clipped =
          entry.raw.cwiseMax(entry.clipMin).cwiseMin(entry.clipMax);
      if (!entry.filterInitialized || entry.lpfCutoffHz < 0.0) {
        entry.filtered = clipped;
        entry.filterInitialized = true;
      } else {
        constexpr double kPi = 3.14159265358979323846;
        const double previousWeight =
            std::exp(-2.0 * kPi * entry.lpfCutoffHz * impl_->policyDt);
        entry.filtered =
            previousWeight * entry.filtered + (1.0 - previousWeight) * clipped;
      }
      entry.processed = entry.term->process(entry.filtered, context);
      if (entry.processed.size() != entry.size())
        throw std::runtime_error("action term '" + entry.name +
                                 "' changed its action size");
      requireFinite(entry.processed,
                    "processed action term '" + entry.name + "'");
    }
    impl_->lastActions.clear();
    for (const auto &entry : impl_->entries) {
      if (entry.term->consumesPolicyOutput())
        impl_->lastActions.emplace(entry.name, entry.processed);
    }
    ++impl_->processSequence;
    processedActions = impl_->lastActions;
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

int ActionManager::build(const ActionFrame &actions, const StepContext &context,
                         InferenceCommand &command, std::string &error) const {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const RobotConfig &robot = env().robotConfig();
    for (const auto &joint : robot.joints) {
      const int index = joint.robot_index;
      command.q(index) = joint.default_position;
      command.dq(index) = joint.default_velocity;
      command.tau(index) = joint.default_torque;
      command.kp(index) = joint.default_kp;
      command.kd(index) = joint.default_kd;
    }
    command.kinematic_modes = robot.kinematic_modes;
    const Eigen::VectorXd noActions;
    for (const std::size_t index : impl_->applyOrder) {
      const auto &entry = impl_->entries[index];
      const Eigen::VectorXd *termActions = &noActions;
      if (entry.term->consumesPolicyOutput()) {
        const auto found = actions.find(entry.name);
        if (found == actions.end())
          throw std::runtime_error("processed actions are missing term: " +
                                   entry.name);
        if (found->second.size() != entry.size())
          throw std::runtime_error("processed action size mismatch for term '" +
                                   entry.name + "': expected " +
                                   std::to_string(entry.size()) + ", got " +
                                   std::to_string(found->second.size()));
        requireFinite(found->second,
                      "processed action term '" + entry.name + "'");
        termActions = &found->second;
      }
      entry.term->apply(*termActions, context, command);
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = "action function failed: " + std::string(exception.what());
    return -1;
  }
}

int ActionManager::termSize(const std::string &termName) const {
  const auto found = impl_->entryIndices.find(termName);
  return found == impl_->entryIndices.end()
             ? 0
             : impl_->entries[found->second].size();
}

std::vector<std::pair<std::string, int>> ActionManager::outputSizes() const {
  std::vector<std::pair<std::string, int>> result;
  result.reserve(impl_->entries.size());
  for (const auto &entry : impl_->entries) {
    if (entry.term->consumesPolicyOutput())
      result.emplace_back(entry.output, entry.size());
  }
  return result;
}

bool ActionManager::getRawActions(const std::string &termName,
                                  Eigen::VectorXd &actions,
                                  uint64_t *sequence) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto found = impl_->entryIndices.find(termName);
  if (found == impl_->entryIndices.end())
    return false;
  actions = impl_->entries[found->second].raw;
  if (sequence != nullptr)
    *sequence = impl_->processSequence;
  return true;
}

bool ActionManager::getProcessedActions(const std::string &termName,
                                        Eigen::VectorXd &actions,
                                        uint64_t *sequence) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto found = impl_->entryIndices.find(termName);
  if (found == impl_->entryIndices.end())
    return false;
  actions = impl_->entries[found->second].processed;
  if (sequence != nullptr)
    *sequence = impl_->processSequence;
  return true;
}

bool ActionManager::getJointMetadata(const std::string &termName,
                                     std::vector<std::string> &jointNames,
                                     Eigen::VectorXd &actionScale) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto found = impl_->entryIndices.find(termName);
  if (found == impl_->entryIndices.end())
    return false;
  const auto *jointTerm = dynamic_cast<const action::JointActionTerm *>(
      impl_->entries[found->second].term.get());
  if (jointTerm == nullptr)
    return false;
  jointNames = jointTerm->configuredJointNames();
  actionScale = jointTerm->configuredActionScale();
  return true;
}

ActionFrame ActionManager::actionFrame() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->lastActions;
}

} // namespace igris_c_inference::policy
