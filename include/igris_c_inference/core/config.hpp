#pragma once

#include "igris_c_inference/core/types.hpp"

#include <Eigen/Dense>

#include <array>
#include <map>
#include <string>
#include <vector>

namespace igris_c_inference {

struct RobotNetworkConfig {
  int domain_id = 0;
  std::string robot_namespace;
  std::string cyclonedds_xml_path;
  std::string state_topic = "rt/lowstate";
  std::string robot_state_topic = "rt/robotstate";
  std::string command_topic = "rt/lowcmd";
};

struct InferenceConfig {
  RobotNetworkConfig robot;
  int control_hz = 300;
  bool update_while_idle = false;
  int state_timeout_ms = 100;
  int first_state_timeout_ms = 5000;
  bool enter_low_level_mode = true;
  int low_level_mode_timeout_ms = 5000;
  int command_timeout_ms = 5000;
  std::string robot_path;
  std::string policy_config_path;
  std::string policy_path;
};

struct RobotJointConfig {
  std::string name;
  std::string model_joint;
  int robot_index = -1;
  double position_min = 0.0;
  double position_max = 0.0;
  double default_position = 0.0;
  double default_velocity = 0.0;
  double default_torque = 0.0;
  double default_kp = 0.0;
  double default_kd = 0.0;
};

struct RobotFrameJointConfig {
  std::string joint;
  int robot_index = -1;
  Eigen::Vector3d axis = Eigen::Vector3d::Zero();
};

struct RobotFrameConfig {
  std::string name;
  std::vector<RobotFrameJointConfig> joints;
};

struct RobotConfig {
  std::array<RobotJointConfig, kModelDof> joints;
  Eigen::Matrix3d base_to_imu_rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d gravity_direction = Eigen::Vector3d(0.0, 0.0, -1.0);
  std::array<KinematicMode, kKinematicGroupCount> kinematic_modes{};
  std::map<std::string, RobotFrameConfig> frames;

  const RobotJointConfig &joint(const std::string &name) const;
  const RobotJointConfig &jointByModelName(const std::string &name) const;
  const RobotFrameConfig &frame(const std::string &name) const;
  std::vector<int>
  resolveJointIndices(const std::vector<std::string> &names) const;
};

InferenceConfig LoadInferenceConfig(const std::string &path);
RobotConfig LoadRobotConfig(const std::string &path);

} // namespace igris_c_inference
