#include "igris_c_inference/modules/policy/managers/ObservationManager.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/mdp/observations/ObservationTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <deque>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace igris_c_inference::policy {
namespace {

enum class HistoryOrder { OLDEST_TO_NEWEST, NEWEST_TO_OLDEST };

} // namespace

class ObservationManager::Impl {
public:
  struct Term {
    std::string name;
    std::string functionName;
    std::unique_ptr<observation::Term> implementation;
    std::size_t historyLength{1};
    std::size_t historyStride{1};
    std::size_t historyCapacity{1};
    HistoryOrder historyOrder{HistoryOrder::OLDEST_TO_NEWEST};
    ObservationUpdatePhase updatePhase{ObservationUpdatePhase::CONTROL};
    double lpfCutoffHz{-1.0};
    bool filterInitialized{false};
    bool historyInitialized{false};
    Eigen::VectorXd filtered;
    Eigen::VectorXd latest;
    Eigen::VectorXd flattened;
    std::deque<Eigen::VectorXd> history;
  };

  struct Group {
    std::string name;
    std::vector<Term> terms;
  };

  int controlHz{0};
  int policyHz{0};
  std::vector<std::string> jointNames;
  Eigen::VectorXd jointReference;
  std::vector<Group> groups;
  bool hasLatest{false};
  mutable std::mutex mutex;

  const Eigen::VectorXd &filter(Term &term,
                                const Eigen::VectorXd &value) const {
    if (!term.filterInitialized || term.lpfCutoffHz < 0.0) {
      term.filtered = value;
      term.filterInitialized = true;
      return term.filtered;
    }
    if (term.filtered.size() != value.size())
      throw std::runtime_error("output size changed: expected " +
                               std::to_string(term.filtered.size()) + ", got " +
                               std::to_string(value.size()));
    constexpr double kPi = 3.14159265358979323846;
    const int hz = term.updatePhase == ObservationUpdatePhase::POLICY
                       ? policyHz
                       : controlHz;
    const double dt = 1.0 / static_cast<double>(hz);
    const double rc = 1.0 / (2.0 * kPi * term.lpfCutoffHz);
    term.filtered += (dt / (rc + dt)) * (value - term.filtered);
    return term.filtered;
  }

  const Eigen::VectorXd &flatten(Term &term) const {
    if (term.history.empty()) {
      term.flattened.resize(0);
      return term.flattened;
    }
    const Eigen::Index width = term.history.front().size();
    term.flattened.resize(width *
                          static_cast<Eigen::Index>(term.historyLength));
    Eigen::Index offset = 0;
    for (std::size_t outputIndex = 0; outputIndex < term.historyLength;
         ++outputIndex) {
      const std::size_t chronological =
          term.historyOrder == HistoryOrder::OLDEST_TO_NEWEST
              ? outputIndex
              : term.historyLength - 1U - outputIndex;
      const std::size_t index =
          term.history.size() - 1U -
          (term.historyLength - 1U - chronological) * term.historyStride;
      term.flattened.segment(offset, width) = term.history[index];
      offset += width;
    }
    return term.flattened;
  }
};

ObservationManager::ObservationManager(RlEnvModule &env)
    : ManagerBase(env), impl_(std::make_unique<Impl>()) {}
ObservationManager::~ObservationManager() = default;

int ObservationManager::configure(const std::string &path, int controlHz,
                                  int policyHz, std::string &error) {
  try {
    if (controlHz <= 0 || policyHz <= 0)
      throw std::runtime_error("control_hz and policy_hz must be positive");
    const YAML::Node root = YAML::LoadFile(path);
    yaml::checkKeys(root, {"igris_c_inference_observations"},
                    "observation root");
    const YAML::Node config = root["igris_c_inference_observations"];
    yaml::checkKeys(config,
                    {"obs_joints", "joint_position_reference", "groups"},
                    "observation");
    impl_->controlHz = controlHz;
    impl_->policyHz = policyHz;
    impl_->jointNames =
        yaml::readNames(config["obs_joints"], "observation.obs_joints");
    env().robotConfig().resolveJointIndices(impl_->jointNames);
    if (config["joint_position_reference"]) {
      impl_->jointReference = yaml::readVector(
          config["joint_position_reference"], impl_->jointNames.size(),
          "observation.joint_position_reference");
    } else {
      impl_->jointReference.resize(
          static_cast<Eigen::Index>(impl_->jointNames.size()));
      for (std::size_t index = 0; index < impl_->jointNames.size(); ++index) {
        impl_->jointReference(static_cast<Eigen::Index>(index)) =
            env()
                .robotConfig()
                .joint(impl_->jointNames[index])
                .default_position;
      }
    }
    const YAML::Node groups = config["groups"];
    if (!groups || !groups.IsSequence() || groups.size() == 0U)
      throw std::runtime_error("observation.groups must be non-empty");
    impl_->groups.clear();
    std::unordered_set<std::string> groupNames;
    const auto &registry = observation::registry();
    const observation::FactoryContext factoryContext{
        env(), &impl_->jointNames, &impl_->jointReference, policyHz,
        std::filesystem::path(path).parent_path().string()};
    for (const auto &groupNode : groups) {
      yaml::checkKeys(groupNode, {"name", "terms"}, "observation.group");
      Impl::Group group;
      group.name =
          yaml::required<std::string>(groupNode, "name", "observation.group");
      if (group.name.empty() || !groupNames.insert(group.name).second)
        throw std::runtime_error("observation group names must be unique");
      const YAML::Node terms = groupNode["terms"];
      if (!terms || !terms.IsSequence() || terms.size() == 0U)
        throw std::runtime_error("observation group terms must be non-empty");
      std::size_t termIndex = 0;
      std::unordered_set<std::string> termNames;
      for (const auto &termNode : terms) {
        yaml::checkKeys(termNode,
                        {"name", "function", "params", "history_length",
                         "history_stride", "history_order", "lpf_cutoff_hz",
                         "update_at"},
                        "observation.term");
        Impl::Term term;
        term.functionName = yaml::required<std::string>(termNode, "function",
                                                        "observation.term");
        term.name = termNode["name"]
                        ? termNode["name"].as<std::string>()
                        : term.functionName + "_" + std::to_string(termIndex);
        ++termIndex;
        if (term.name.empty() || !termNames.insert(term.name).second)
          throw std::runtime_error(
              "observation term names must be non-empty and unique within a "
              "group");
        const auto factory = registry.find(term.functionName);
        if (factory == registry.end())
          throw std::runtime_error("unknown observation function: " +
                                   term.functionName);
        const YAML::Node params =
            yaml::optionalMap(termNode, "params", "observation.term");
        term.implementation =
            factory->second(term.name, params, factoryContext);
        if (!term.implementation)
          throw std::runtime_error("observation factory returned null: " +
                                   term.functionName);
        const int history = termNode["history_length"]
                                ? termNode["history_length"].as<int>()
                                : 1;
        const int stride = termNode["history_stride"]
                               ? termNode["history_stride"].as<int>()
                               : 1;
        if (history <= 0 || stride <= 0)
          throw std::runtime_error(
              "observation history length and stride must be positive");
        term.historyLength = static_cast<std::size_t>(history);
        term.historyStride = static_cast<std::size_t>(stride);
        term.historyCapacity =
            1U + (term.historyLength - 1U) * term.historyStride;
        const std::string order =
            termNode["history_order"]
                ? termNode["history_order"].as<std::string>()
                : "oldest_to_newest";
        if (order == "oldest_to_newest")
          term.historyOrder = HistoryOrder::OLDEST_TO_NEWEST;
        else if (order == "newest_to_oldest")
          term.historyOrder = HistoryOrder::NEWEST_TO_OLDEST;
        else
          throw std::runtime_error(
              "history_order must be oldest_to_newest or newest_to_oldest");
        const std::string updateAt =
            termNode["update_at"] ? termNode["update_at"].as<std::string>()
                                  : "control";
        if (updateAt == "control")
          term.updatePhase = ObservationUpdatePhase::CONTROL;
        else if (updateAt == "policy")
          term.updatePhase = ObservationUpdatePhase::POLICY;
        else
          throw std::runtime_error("update_at must be control or policy");
        term.lpfCutoffHz = termNode["lpf_cutoff_hz"]
                               ? termNode["lpf_cutoff_hz"].as<double>()
                               : -1.0;
        if (!std::isfinite(term.lpfCutoffHz) ||
            (term.lpfCutoffHz <= 0.0 && term.lpfCutoffHz != -1.0))
          throw std::runtime_error("lpf_cutoff_hz must be -1 or positive");
        group.terms.push_back(std::move(term));
      }
      impl_->groups.push_back(std::move(group));
    }
    reset();
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

void ObservationManager::reset() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (auto &group : impl_->groups) {
    for (auto &term : group.terms) {
      term.implementation->reset();
      term.filterInitialized = false;
      term.historyInitialized = false;
      term.filtered.resize(0);
      term.latest.resize(0);
      term.flattened.resize(0);
      term.history.clear();
    }
  }
  impl_->hasLatest = false;
}

std::vector<TermDescriptor> ObservationManager::termDescriptors() const {
  std::vector<TermDescriptor> result;
  for (const auto &group : impl_->groups) {
    for (const auto &term : group.terms) {
      const int size =
          term.implementation->size() * static_cast<int>(term.historyLength);
      result.push_back(TermDescriptor{
          group.name + "." + term.name,
          term.functionName,
          group.name,
          term.updatePhase == ObservationUpdatePhase::CONTROL ? "control"
                                                              : "policy",
          size,
          {},
          {}});
    }
  }
  return result;
}

int ObservationManager::update(const StepContext &context,
                               ObservationUpdatePhase phase,
                               std::string &error) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto &group : impl_->groups) {
      for (auto &term : group.terms) {
        if (term.updatePhase != phase)
          continue;
        try {
          const Eigen::VectorXd value = term.implementation->compute(context);
          if (value.size() != term.implementation->size())
            throw std::runtime_error(
                "returned " + std::to_string(value.size()) +
                " values, expected " +
                std::to_string(term.implementation->size()));
          for (Eigen::Index index = 0; index < value.size(); ++index) {
            if (!std::isfinite(value(index)))
              throw std::runtime_error("returned a non-finite value at index " +
                                       std::to_string(index));
          }
          term.latest = impl_->filter(term, value);
        } catch (const std::exception &exception) {
          throw std::runtime_error("group '" + group.name + "', term '" +
                                   term.functionName +
                                   "': " + exception.what());
        }
      }
    }
    impl_->hasLatest = true;
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

std::vector<std::pair<std::string, int>>
ObservationManager::groupSizes() const {
  std::vector<std::pair<std::string, int>> result;
  result.reserve(impl_->groups.size());
  for (const auto &group : impl_->groups) {
    int size = 0;
    for (const auto &term : group.terms) {
      size +=
          term.implementation->size() * static_cast<int>(term.historyLength);
    }
    result.emplace_back(group.name, size);
  }
  return result;
}

int ObservationManager::buildLatest(TensorMap &groups, std::string &error) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->hasLatest)
      throw std::runtime_error("observations have not been updated");
    groups.clear();
    for (auto &group : impl_->groups) {
      Eigen::Index totalSize = 0;
      for (auto &term : group.terms) {
        if (term.latest.size() == 0)
          throw std::runtime_error("group '" + group.name + "', term '" +
                                   term.functionName +
                                   "' has not been updated");
        if (!term.historyInitialized) {
          term.history.assign(term.historyCapacity, term.latest);
          term.historyInitialized = true;
        } else {
          if (term.history.size() == term.historyCapacity)
            term.history.pop_front();
          term.history.push_back(term.latest);
        }
        totalSize += impl_->flatten(term).size();
      }
      Eigen::VectorXd &flattened = groups[group.name];
      flattened.resize(totalSize);
      Eigen::Index offset = 0;
      for (const auto &term : group.terms) {
        flattened.segment(offset, term.flattened.size()) = term.flattened;
        offset += term.flattened.size();
      }
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

} // namespace igris_c_inference::policy
