#pragma once

#include <Eigen/Geometry>

#include <cmath>
#include <stdexcept>
#include <string>

namespace igris_c_inference {

class ReferenceHeadingAlignment {
public:
  void configure(const std::string &mode, const std::string &where) {
    if (mode == "raw")
      mode_ = Mode::Raw;
    else if (mode == "initial")
      mode_ = Mode::Initial;
    else if (mode == "remove_yaw")
      mode_ = Mode::RemoveYaw;
    else
      throw std::runtime_error(
          where + " must be 'raw', 'initial', or 'remove_yaw', got '" + mode +
          "'");
    reset();
  }

  void reset() {
    latched_ = false;
    world_heading_offset_.setIdentity();
  }

  bool requiresLatch() const { return mode_ == Mode::Initial && !latched_; }

  Eigen::Matrix3d relative(const Eigen::Matrix3d &world_from_robot,
                           const Eigen::Matrix3d &world_from_reference) {
    if (!world_from_robot.allFinite() || !world_from_reference.allFinite())
      throw std::runtime_error(
          "reference heading alignment received a non-finite rotation");
    if (mode_ == Mode::Raw)
      return world_from_robot.transpose() * world_from_reference;
    if (mode_ == Mode::RemoveYaw)
      return withoutYaw(world_from_robot).transpose() *
             withoutYaw(world_from_reference);
    if (!latched_) {
      const double robot_yaw =
          std::atan2(world_from_robot(1, 0), world_from_robot(0, 0));
      const double reference_yaw = std::atan2(world_from_reference(1, 0),
                                              world_from_reference(0, 0));
      const double yaw = robot_yaw - reference_yaw;
      if (!std::isfinite(yaw))
        throw std::runtime_error(
            "reference heading alignment produced a non-finite yaw");
      world_heading_offset_ =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
      latched_ = true;
    }
    return world_from_robot.transpose() * world_heading_offset_ *
           world_from_reference;
  }

private:
  // Match igris_c_gmt_public's RemoveYawFromQuaternion: remove each world
  // heading before computing the robot-to-reference relative rotation.
  static Eigen::Matrix3d withoutYaw(const Eigen::Matrix3d &world_from_body) {
    const Eigen::Quaterniond quaternion(world_from_body);
    const Eigen::Quaterniond normalized = quaternion.normalized();
    const double yaw =
        std::atan2(2.0 * (normalized.w() * normalized.z() +
                          normalized.x() * normalized.y()),
                   1.0 - 2.0 * (normalized.y() * normalized.y() +
                                normalized.z() * normalized.z()));
    const Eigen::Quaterniond heading(std::cos(0.5 * yaw), 0.0, 0.0,
                                     std::sin(0.5 * yaw));
    return (heading.conjugate() * normalized).normalized().toRotationMatrix();
  }

  enum class Mode { Raw, Initial, RemoveYaw };

  Mode mode_ = Mode::Raw;
  bool latched_ = false;
  Eigen::Matrix3d world_heading_offset_ = Eigen::Matrix3d::Identity();
};

} // namespace igris_c_inference
