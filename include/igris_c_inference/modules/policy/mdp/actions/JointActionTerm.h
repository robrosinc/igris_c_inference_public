#pragma once

#include "igris_c_inference/core/config.hpp"
#include "igris_c_inference/modules/policy/mdp/actions/ActionTerm.h"

#include <array>
#include <string>
#include <vector>

namespace igris_c_inference::policy::action {

class JointActionTerm : public Term {
public:
  JointActionTerm(std::string name, RlEnvModule &env, const YAML::Node &config);

  int actionSize() const override;
  void applyKinematicModes(InferenceCommand &command) const;
  const std::vector<std::string> &configuredJointNames() const {
    return joint_names_;
  }
  const Eigen::VectorXd &configuredActionScale() const { return action_scale_; }

protected:
  const RobotConfig &robot() const;
  const std::vector<std::string> &jointNames() const { return joint_names_; }
  const std::vector<int> &jointIndices() const { return joint_indices_; }
  const Eigen::VectorXd &defaultPositions() const { return default_positions_; }
  const Eigen::VectorXd &actionScale() const { return action_scale_; }
  const Eigen::VectorXd &kp() const { return kp_; }
  const Eigen::VectorXd &kd() const { return kd_; }
  const Eigen::VectorXd &positionMin() const { return position_min_; }
  const Eigen::VectorXd &positionMax() const { return position_max_; }

private:
  std::vector<std::string> joint_names_;
  std::vector<int> joint_indices_;
  Eigen::VectorXd default_positions_;
  Eigen::VectorXd action_scale_;
  Eigen::VectorXd kp_;
  Eigen::VectorXd kd_;
  Eigen::VectorXd position_min_;
  Eigen::VectorXd position_max_;
  std::array<KinematicMode, kKinematicGroupCount> modes_{};
};

} // namespace igris_c_inference::policy::action
