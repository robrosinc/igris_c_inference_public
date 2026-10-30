#include "igris_c_inference/modules/state_handler.hpp"

#include <Eigen/Geometry>

#include <cmath>

namespace igris_c_inference {

void StateHandler::configure(const RobotConfig &robot) { robot_ = robot; }

bool StateHandler::update(const igris_c_sdk::msg::dds_::LowState_ &raw,
                          uint64_t sequence, ProcessedState &state) const {
  for (int i = 0; i < kModelDof; ++i) {
    state.joint_position(i) =
        static_cast<double>(raw.joint_state()[static_cast<std::size_t>(i)].q());
    state.joint_velocity(i) = static_cast<double>(
        raw.joint_state()[static_cast<std::size_t>(i)].dq());
  }
  const auto &orientation = raw.imu_state().orientation();
  Eigen::Quaterniond world_from_imu(static_cast<double>(orientation.w()),
                                    static_cast<double>(orientation.x()),
                                    static_cast<double>(orientation.y()),
                                    static_cast<double>(orientation.z()));
  if (!std::isfinite(world_from_imu.norm()) || world_from_imu.norm() <= 1.0e-12)
    return false;
  world_from_imu.normalize();
  const Eigen::Matrix3d world_from_imu_rotation =
      world_from_imu.toRotationMatrix();
  const Eigen::Matrix3d world_from_base =
      world_from_imu_rotation * robot_.base_to_imu_rotation;
  const auto &gyro = raw.imu_state().angular_velocity();
  const Eigen::Vector3d imu_gyro(static_cast<double>(gyro.x()),
                                 static_cast<double>(gyro.y()),
                                 static_cast<double>(gyro.z()));
  state.base_angular_velocity =
      world_from_base.transpose() * world_from_imu_rotation * imu_gyro;
  state.world_from_base = world_from_base;
  state.base_angular_velocity_world = world_from_imu_rotation * imu_gyro;
  state.projected_gravity =
      world_from_base.transpose() * robot_.gravity_direction;
  state.sequence = sequence;
  state.valid = state.joint_position.allFinite() &&
                state.joint_velocity.allFinite() &&
                state.base_angular_velocity.allFinite() &&
                state.projected_gravity.allFinite() &&
                state.world_from_base.allFinite() &&
                state.base_angular_velocity_world.allFinite();
  return state.valid;
}

} // namespace igris_c_inference
