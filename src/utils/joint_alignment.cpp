#include "igris_c_inference/utils/joint_alignment.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace igris_c_inference {
namespace {

std::vector<std::string> ReadNames(const YAML::Node &node,
                                   const std::string &where) {
  if (!node || !node.IsSequence() || node.size() == 0)
    throw std::runtime_error(where + " must be a non-empty sequence");
  std::vector<std::string> output;
  std::unordered_set<std::string> seen;
  for (const auto &value : node) {
    const std::string name = value.as<std::string>();
    if (name.empty() || !seen.emplace(name).second)
      throw std::runtime_error(where + " contains an empty or duplicate name");
    output.push_back(name);
  }
  return output;
}

Eigen::VectorXd ReadNamedVector(const YAML::Node &node,
                                const std::vector<std::string> &names,
                                const std::string &where) {
  if (!node || !node.IsMap())
    throw std::runtime_error(where + " must be a map keyed by joint name");
  Eigen::VectorXd output(static_cast<Eigen::Index>(names.size()));
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (!node[names[i]])
      throw std::runtime_error(where + " is missing '" + names[i] + "'");
    output(static_cast<Eigen::Index>(i)) = node[names[i]].as<double>();
  }
  if (node.size() != names.size() || !output.allFinite())
    throw std::runtime_error(where + " must exactly cover finite joint values");
  return output;
}

const RobotJointConfig &ResolveRobotJoint(const RobotConfig &robot,
                                          const std::string &name) {
  try {
    return robot.joint(name);
  } catch (const std::exception &) {
    return robot.jointByModelName(name);
  }
}

} // namespace

void JointAlignment::configure(const std::string &path,
                               const RobotConfig &robot) {
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node alignment = root["alignment"];
  if (!alignment || !alignment.IsMap())
    throw std::runtime_error("alignment file must contain an alignment map");
  const std::string mode =
      alignment["joint_position_mode"]
          ? alignment["joint_position_mode"].as<std::string>()
          : "relative";
  if (mode == "absolute")
    position_mode_ = PositionMode::Absolute;
  else if (mode == "relative")
    position_mode_ = PositionMode::Relative;
  else
    throw std::runtime_error(
        "alignment.joint_position_mode must be absolute or relative");

  source_joint_names_ = ReadNames(alignment["source_joint_names"],
                                  "alignment.source_joint_names");
  target_joint_names_ = ReadNames(alignment["target_joint_names"],
                                  "alignment.target_joint_names");
  std::unordered_map<std::string, std::size_t> source_index;
  std::unordered_map<std::string, std::size_t> target_index;
  for (std::size_t i = 0; i < source_joint_names_.size(); ++i)
    source_index.emplace(source_joint_names_[i], i);
  for (std::size_t i = 0; i < target_joint_names_.size(); ++i)
    target_index.emplace(target_joint_names_[i], i);

  source_from_target_ = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(source_joint_names_.size()),
      static_cast<Eigen::Index>(target_joint_names_.size()));
  std::unordered_set<std::string> mapped_source;
  std::unordered_set<std::string> mapped_target;
  const YAML::Node mapping = alignment["joint_mapping"];
  if (!mapping || !mapping.IsSequence())
    throw std::runtime_error("alignment.joint_mapping must be a sequence");
  for (const auto &pair : mapping) {
    const std::string source = pair["source"].as<std::string>();
    const std::string target = pair["target"].as<std::string>();
    const double sign = pair["sign"] ? pair["sign"].as<double>() : 1.0;
    if (source_index.find(source) == source_index.end() ||
        target_index.find(target) == target_index.end())
      throw std::runtime_error("joint mapping contains an unknown joint");
    if (!mapped_source.emplace(source).second ||
        !mapped_target.emplace(target).second)
      throw std::runtime_error("joint mapping contains a duplicate joint");
    if (sign != -1.0 && sign != 1.0)
      throw std::runtime_error("joint mapping sign must be +1 or -1");
    source_from_target_(static_cast<Eigen::Index>(source_index.at(source)),
                        static_cast<Eigen::Index>(target_index.at(target))) =
        sign;
  }
  if (mapped_target.size() != target_joint_names_.size())
    throw std::runtime_error("every target joint must be mapped exactly once");

  target_robot_indices_.clear();
  target_short_names_.clear();
  target_default_.resize(static_cast<Eigen::Index>(target_joint_names_.size()));
  for (std::size_t i = 0; i < target_joint_names_.size(); ++i) {
    const auto &joint = ResolveRobotJoint(robot, target_joint_names_[i]);
    target_robot_indices_.push_back(joint.robot_index);
    target_short_names_.push_back(joint.name);
    target_default_(static_cast<Eigen::Index>(i)) = joint.default_position;
  }
  source_default_ =
      ReadNamedVector(root["source_default_joint_pos"], source_joint_names_,
                      "source_default_joint_pos");
  source_action_scale_ = ReadNamedVector(
      root["source_action_scale"], source_joint_names_, "source_action_scale");
  if ((source_action_scale_.array() <= 0.0).any())
    throw std::runtime_error("source_action_scale must be positive");
}

Eigen::VectorXd JointAlignment::gatherTarget(const VectorQ &values) const {
  Eigen::VectorXd output(static_cast<Eigen::Index>(targetDofs()));
  for (std::size_t i = 0; i < targetDofs(); ++i)
    output(static_cast<Eigen::Index>(i)) = values(target_robot_indices_[i]);
  return output;
}

Eigen::VectorXd
JointAlignment::reorderTarget(const Eigen::VectorXd &values,
                              const std::vector<std::string> &names) const {
  if (values.size() != static_cast<Eigen::Index>(names.size()))
    throw std::runtime_error("joint values and names have different sizes");
  std::unordered_map<std::string, Eigen::Index> input;
  for (std::size_t i = 0; i < names.size(); ++i) {
    const std::string &name = names[i];
    if (!input.emplace(name, static_cast<Eigen::Index>(i)).second)
      throw std::runtime_error("input joint names contain duplicates");
  }
  Eigen::VectorXd output(static_cast<Eigen::Index>(targetDofs()));
  for (std::size_t i = 0; i < targetDofs(); ++i) {
    auto found = input.find(target_joint_names_[i]);
    if (found == input.end())
      found = input.find(target_short_names_[i]);
    if (found == input.end()) {
      throw std::runtime_error("input joint vector is missing '" +
                               target_joint_names_[i] + "'");
    }
    output(static_cast<Eigen::Index>(i)) = values(found->second);
  }
  return output;
}

Eigen::VectorXd JointAlignment::sourcePosition(const VectorQ &target_position,
                                               bool output_relative) const {
  const Eigen::VectorXd target = gatherTarget(target_position);
  Eigen::VectorXd source;
  if (position_mode_ == PositionMode::Absolute)
    source = source_from_target_ * target;
  else
    source = source_from_target_ * (target - target_default_) + source_default_;
  return output_relative ? source - source_default_ : source;
}

Eigen::VectorXd
JointAlignment::sourcePosition(const Eigen::VectorXd &target_position,
                               const std::vector<std::string> &names,
                               bool output_relative) const {
  const Eigen::VectorXd target = reorderTarget(target_position, names);
  Eigen::VectorXd source;
  if (position_mode_ == PositionMode::Absolute)
    source = source_from_target_ * target;
  else
    source = source_from_target_ * (target - target_default_) + source_default_;
  return output_relative ? source - source_default_ : source;
}

Eigen::VectorXd
JointAlignment::sourceVelocity(const VectorQ &target_velocity) const {
  return source_from_target_ * gatherTarget(target_velocity);
}

Eigen::VectorXd
JointAlignment::sourceVelocity(const Eigen::VectorXd &target_velocity,
                               const std::vector<std::string> &names) const {
  return source_from_target_ * reorderTarget(target_velocity, names);
}

Eigen::VectorXd
JointAlignment::sourceAction(const Eigen::VectorXd &target_action,
                             const std::vector<std::string> &action_names,
                             const Eigen::VectorXd &target_scale) const {
  if (target_action.size() != target_scale.size())
    throw std::runtime_error("target action and scale sizes differ");
  const Eigen::VectorXd physical = target_action.cwiseProduct(target_scale);
  return (source_from_target_ * reorderTarget(physical, action_names))
      .cwiseQuotient(source_action_scale_);
}

} // namespace igris_c_inference
