#pragma once

#include "igris_c_sdk/igris_c_msgs.hpp"
#include "igris_c_sdk/types.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstdint>

namespace igris_c_inference {

inline constexpr int kModelDof = static_cast<int>(igris_c_sdk::N_JOINTS);
inline constexpr int kKinematicGroupCount = 5;

using VectorQ = Eigen::Matrix<double, kModelDof, 1>;
using KinematicMode = uint8_t;

struct InferenceCommand {
  std::array<KinematicMode, kKinematicGroupCount> kinematic_modes{};
  VectorQ q = VectorQ::Zero();
  VectorQ dq = VectorQ::Zero();
  VectorQ tau = VectorQ::Zero();
  VectorQ kp = VectorQ::Zero();
  VectorQ kd = VectorQ::Zero();
};

} // namespace igris_c_inference
