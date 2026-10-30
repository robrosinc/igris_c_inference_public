#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/managers/ActionManager.h"
#include "igris_c_inference/modules/policy/mdp/observations/ObservationTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"
#include "igris_c_inference/utils/joint_alignment.hpp"

#include <Eigen/Geometry>

#include <filesystem>
#include <memory>
#include <stdexcept>

namespace igris_c_inference::policy::observation {
namespace {

std::string resolvePath(const FactoryContext &context,
                        const std::string &value) {
  const std::filesystem::path path(value);
  return (path.is_absolute()
              ? path
              : std::filesystem::path(context.profileDirectory) / path)
      .lexically_normal()
      .string();
}

Eigen::Matrix3d worldFromFrame(const ProcessedState &status,
                               const RobotFrameConfig &frame) {
  Eigen::Matrix3d rotation = status.world_from_base;
  for (const auto &joint : frame.joints) {
    rotation *=
        Eigen::AngleAxisd(status.joint_position(joint.robot_index), joint.axis)
            .toRotationMatrix();
  }
  return rotation;
}

std::shared_ptr<JointAlignment> loadAlignment(const YAML::Node &params,
                                              const FactoryContext &context,
                                              const std::string &function) {
  yaml::checkKeys(params, {"alignment", "name"}, "observation.term.params");
  auto alignment = std::make_shared<JointAlignment>();
  alignment->configure(
      resolvePath(context, yaml::required<std::string>(
                               params, "alignment", "observation.term.params")),
      context.env.robotConfig());
  if (function != "aligned_last_actions" && params["name"])
    throw std::runtime_error(function + " does not accept params.name");
  return alignment;
}

} // namespace

std::unique_ptr<Term>
createFrameAngularVelocityObservation(const std::string &name,
                                      const YAML::Node &params,
                                      const FactoryContext &context) {
  yaml::checkKeys(params, {"frame_name"}, "observation.term.params");
  const RobotFrameConfig frame =
      context.env.robotConfig().frame(yaml::required<std::string>(
          params, "frame_name", "observation.term.params"));
  return makeFunctionTerm(
      name, context.env, 3, [frame](RlEnvModule &, const StepContext &step) {
        Eigen::Matrix3d rotation = step.status.world_from_base;
        Eigen::Vector3d angular = step.status.base_angular_velocity_world;
        for (const auto &joint : frame.joints) {
          angular += rotation * joint.axis *
                     step.status.joint_velocity(joint.robot_index);
          rotation *=
              Eigen::AngleAxisd(step.status.joint_position(joint.robot_index),
                                joint.axis)
                  .toRotationMatrix();
        }
        return Eigen::VectorXd(rotation.transpose() * angular);
      });
}

std::unique_ptr<Term>
createFrameProjectedGravityObservation(const std::string &name,
                                       const YAML::Node &params,
                                       const FactoryContext &context) {
  yaml::checkKeys(params, {"frame_name"}, "observation.term.params");
  const RobotFrameConfig frame =
      context.env.robotConfig().frame(yaml::required<std::string>(
          params, "frame_name", "observation.term.params"));
  const Eigen::Vector3d gravity = context.env.robotConfig().gravity_direction;
  return makeFunctionTerm(
      name, context.env, 3,
      [frame, gravity](RlEnvModule &, const StepContext &step) {
        return Eigen::VectorXd(worldFromFrame(step.status, frame).transpose() *
                               gravity);
      });
}

std::unique_ptr<Term>
createAlignedJointPositionObservation(const std::string &name,
                                      const YAML::Node &params,
                                      const FactoryContext &context) {
  auto alignment = loadAlignment(params, context, "aligned_joint_pos_rel");
  return makeFunctionTerm(
      name, context.env, static_cast<int>(alignment->sourceDofs()),
      [alignment](RlEnvModule &, const StepContext &step) {
        return alignment->sourcePosition(step.status.joint_position, true);
      });
}

std::unique_ptr<Term>
createAlignedJointVelocityObservation(const std::string &name,
                                      const YAML::Node &params,
                                      const FactoryContext &context) {
  auto alignment = loadAlignment(params, context, "aligned_joint_vel");
  return makeFunctionTerm(
      name, context.env, static_cast<int>(alignment->sourceDofs()),
      [alignment](RlEnvModule &, const StepContext &step) {
        return alignment->sourceVelocity(step.status.joint_velocity);
      });
}

std::unique_ptr<Term>
createAlignedLastActionsObservation(const std::string &name,
                                    const YAML::Node &params,
                                    const FactoryContext &context) {
  auto alignment = loadAlignment(params, context, "aligned_last_actions");
  const std::string actionName =
      yaml::required<std::string>(params, "name", "observation.term.params");
  std::vector<std::string> actionNames;
  Eigen::VectorXd actionScale;
  if (!context.env.actionManager().getJointMetadata(actionName, actionNames,
                                                    actionScale))
    throw std::runtime_error(
        "aligned_last_actions references unknown or non-joint action term: " +
        actionName);
  return makeFunctionTerm(
      name, context.env, static_cast<int>(alignment->sourceDofs()),
      [alignment, actionName, actionNames, actionScale](RlEnvModule &env,
                                                        const StepContext &) {
        Eigen::VectorXd actions;
        if (!env.actionManager().getRawActions(actionName, actions))
          throw std::runtime_error(
              "aligned_last_actions cannot read action term: " + actionName);
        return alignment->sourceAction(actions, actionNames, actionScale);
      });
}

} // namespace igris_c_inference::policy::observation
