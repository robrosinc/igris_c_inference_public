#pragma once

#include "igris_c_inference/core/types.hpp"

#include <Eigen/Dense>

#include <cstdint>

namespace igris_c_inference {

struct ProcessedState {
  bool valid = false;
  uint64_t sequence = 0;
  VectorQ joint_position = VectorQ::Zero();
  VectorQ joint_velocity = VectorQ::Zero();
  Eigen::Vector3d base_angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d projected_gravity = Eigen::Vector3d::Zero();
  Eigen::Matrix3d world_from_base = Eigen::Matrix3d::Identity();
  Eigen::Vector3d base_angular_velocity_world = Eigen::Vector3d::Zero();
};

} // namespace igris_c_inference
