#pragma once

#include "igris_c_inference/core/config.hpp"

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace igris_c_inference {

// Name-driven linear map from the deployed robot coordinates to the source
// policy coordinates. Unmapped source joints (the G1 wrists for SONIC) stay 0.
class JointAlignment {
public:
  enum class PositionMode { Absolute, Relative };

  void configure(const std::string &path, const RobotConfig &robot);

  Eigen::VectorXd sourcePosition(const VectorQ &target_position,
                                 bool output_relative) const;
  Eigen::VectorXd sourcePosition(const Eigen::VectorXd &target_position,
                                 const std::vector<std::string> &names,
                                 bool output_relative) const;
  Eigen::VectorXd sourceVelocity(const VectorQ &target_velocity) const;
  Eigen::VectorXd sourceVelocity(const Eigen::VectorXd &target_velocity,
                                 const std::vector<std::string> &names) const;
  Eigen::VectorXd sourceAction(const Eigen::VectorXd &target_action,
                               const std::vector<std::string> &action_names,
                               const Eigen::VectorXd &target_scale) const;

  std::size_t sourceDofs() const { return source_joint_names_.size(); }
  std::size_t targetDofs() const { return target_joint_names_.size(); }
  const std::vector<std::string> &sourceJointNames() const {
    return source_joint_names_;
  }
  const std::vector<std::string> &targetJointNames() const {
    return target_joint_names_;
  }
  PositionMode positionMode() const { return position_mode_; }

private:
  Eigen::VectorXd gatherTarget(const VectorQ &values) const;
  Eigen::VectorXd reorderTarget(const Eigen::VectorXd &values,
                                const std::vector<std::string> &names) const;

  PositionMode position_mode_ = PositionMode::Relative;
  std::vector<std::string> source_joint_names_;
  std::vector<std::string> target_joint_names_;
  std::vector<std::string> target_short_names_;
  std::vector<int> target_robot_indices_;
  Eigen::MatrixXd source_from_target_;
  Eigen::VectorXd source_default_;
  Eigen::VectorXd source_action_scale_;
  Eigen::VectorXd target_default_;
};

} // namespace igris_c_inference
