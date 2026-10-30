#pragma once

#include "igris_c_inference/modules/policy/mdp/commands/CommandTerm.h"

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace igris_c_inference::policy::command {

struct MotionReferenceFrame {
  uint64_t sequence = 0;
  std::vector<std::string> joint_names;
  Eigen::VectorXd joint_position;
  Eigen::VectorXd joint_velocity;
  std::vector<std::string> body_names;
  std::vector<Eigen::Vector3d> anchor_from_body_position;
  std::vector<Eigen::Quaterniond> world_from_body;
  std::string anchor_body_name;
  Eigen::Vector3d world_anchor_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d anchor_linear_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d anchor_angular_velocity = Eigen::Vector3d::Zero();
  bool world_anchor_position_valid = false;
  bool anchor_linear_velocity_valid = false;
  bool anchor_angular_velocity_valid = false;
};

class MotionReferenceSource {
public:
  virtual ~MotionReferenceSource() = default;
  virtual bool start() = 0;
  virtual void stop() = 0;
  virtual void reset() = 0;
  virtual bool waitUntilReady(std::size_t frame_count,
                              std::chrono::milliseconds timeout) = 0;
  virtual bool frames(uint64_t policy_step, std::size_t frame_count,
                      std::vector<MotionReferenceFrame> &output) = 0;
  virtual bool stale(Clock::time_point now,
                     std::chrono::milliseconds maximum_age) const = 0;
};

std::unique_ptr<MotionReferenceSource>
createMotionReferenceSource(const YAML::Node &source,
                            const FactoryContext &context,
                            const std::string &where);

} // namespace igris_c_inference::policy::command
