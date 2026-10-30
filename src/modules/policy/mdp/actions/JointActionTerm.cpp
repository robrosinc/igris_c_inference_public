#include "igris_c_inference/modules/policy/mdp/actions/JointActionTerm.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <stdexcept>
#include <utility>

namespace igris_c_inference::policy::action {

JointActionTerm::JointActionTerm(std::string name, RlEnvModule &env,
                                 const YAML::Node &config)
    : Term(std::move(name), env) {
  const RobotConfig &robot = env.robotConfig();
  const std::string where = "actions.terms." + this->name();
  joint_names_ =
      yaml::readNames(config["action_joints"], where + ".action_joints");
  joint_indices_ = robot.resolveJointIndices(joint_names_);
  const std::size_t size = joint_names_.size();
  action_scale_ =
      yaml::readVector(config["action_scale"], size, where + ".action_scale");
  kp_ = yaml::readVector(config["kp"], size, where + ".kp");
  kd_ = yaml::readVector(config["kd"], size, where + ".kd");
  if ((kp_.array() < 0.0).any() || (kd_.array() < 0.0).any())
    throw std::runtime_error(where + " gains must not be negative");

  default_positions_.resize(static_cast<Eigen::Index>(size));
  position_min_.resize(static_cast<Eigen::Index>(size));
  position_max_.resize(static_cast<Eigen::Index>(size));
  for (std::size_t index = 0; index < size; ++index) {
    const RobotJointConfig &joint = robot.joint(joint_names_[index]);
    const Eigen::Index value_index = static_cast<Eigen::Index>(index);
    default_positions_(value_index) = joint.default_position;
    position_min_(value_index) = joint.position_min;
    position_max_(value_index) = joint.position_max;
  }
  modes_ = robot.kinematic_modes;
}

int JointActionTerm::actionSize() const {
  return static_cast<int>(joint_indices_.size());
}

void JointActionTerm::applyKinematicModes(InferenceCommand &command) const {
  command.kinematic_modes = modes_;
}

const RobotConfig &JointActionTerm::robot() const {
  return env().robotConfig();
}

} // namespace igris_c_inference::policy::action
