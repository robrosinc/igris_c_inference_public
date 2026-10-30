#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/managers/ActionManager.h"
#include "igris_c_inference/modules/policy/mdp/observations/ObservationTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <algorithm>
#include <stdexcept>

namespace igris_c_inference::policy::observation {
namespace {

std::vector<std::string> selectedJointNames(const YAML::Node &params,
                                            const FactoryContext &context) {
  if (context.observationJointNames == nullptr)
    throw std::runtime_error("observation factory context has no joint names");
  if (params["joints"])
    return yaml::readNames(params["joints"], "observation.term.params.joints");
  return *context.observationJointNames;
}

} // namespace

std::unique_ptr<Term>
createBaseAngularVelocityObservation(const std::string &name,
                                     const YAML::Node &params,
                                     const FactoryContext &context) {
  yaml::checkKeys(params, {}, "observation.term.params");
  return makeFunctionTerm(
      name, context.env, 3, [](RlEnvModule &, const StepContext &step) {
        return Eigen::VectorXd(step.status.base_angular_velocity);
      });
}

std::unique_ptr<Term>
createProjectedGravityObservation(const std::string &name,
                                  const YAML::Node &params,
                                  const FactoryContext &context) {
  yaml::checkKeys(params, {}, "observation.term.params");
  return makeFunctionTerm(
      name, context.env, 3, [](RlEnvModule &, const StepContext &step) {
        return Eigen::VectorXd(step.status.projected_gravity);
      });
}

std::unique_ptr<Term>
createLastActionsObservation(const std::string &name, const YAML::Node &params,
                             const FactoryContext &context) {
  yaml::checkKeys(params, {"name"}, "observation.term.params");
  const std::string actionName =
      yaml::required<std::string>(params, "name", "observation.term.params");
  const int size = context.env.actionManager().termSize(actionName);
  if (size <= 0)
    throw std::runtime_error(
        "last_actions references unknown or output-free action term: " +
        actionName);
  return makeFunctionTerm(
      name, context.env, size,
      [actionName](RlEnvModule &env, const StepContext &) {
        Eigen::VectorXd actions;
        if (!env.actionManager().getRawActions(actionName, actions))
          throw std::runtime_error("last_actions cannot read action term: " +
                                   actionName);
        return actions;
      });
}

std::unique_ptr<Term>
createRelativeJointPositionObservation(const std::string &name,
                                       const YAML::Node &params,
                                       const FactoryContext &context) {
  yaml::checkKeys(params, {"joints", "reference"}, "observation.term.params");
  if (context.observationJointReference == nullptr ||
      context.observationJointNames == nullptr)
    throw std::runtime_error(
        "joint_pos_rel factory context has no joint reference");
  const std::vector<std::string> names = selectedJointNames(params, context);
  const std::vector<int> indices =
      context.env.robotConfig().resolveJointIndices(names);
  Eigen::VectorXd reference;
  if (params["reference"]) {
    reference = yaml::readVector(params["reference"], names.size(),
                                 "observation.term.params.reference");
  } else {
    reference.resize(static_cast<Eigen::Index>(names.size()));
    for (std::size_t index = 0; index < names.size(); ++index) {
      const auto found =
          std::find(context.observationJointNames->begin(),
                    context.observationJointNames->end(), names[index]);
      if (found == context.observationJointNames->end())
        throw std::runtime_error(
            "joint_pos_rel requires params.reference for joints outside "
            "obs_joints");
      reference(static_cast<Eigen::Index>(index)) =
          (*context.observationJointReference)(
              std::distance(context.observationJointNames->begin(), found));
    }
  }
  return makeFunctionTerm(
      name, context.env, static_cast<int>(indices.size()),
      [indices, reference](RlEnvModule &, const StepContext &step) {
        Eigen::VectorXd result(static_cast<Eigen::Index>(indices.size()));
        for (std::size_t index = 0; index < indices.size(); ++index)
          result(static_cast<Eigen::Index>(index)) =
              step.status.joint_position(indices[index]) -
              reference(static_cast<Eigen::Index>(index));
        return result;
      });
}

std::unique_ptr<Term>
createJointVelocityObservation(const std::string &name,
                               const YAML::Node &params,
                               const FactoryContext &context) {
  yaml::checkKeys(params, {"joints"}, "observation.term.params");
  const std::vector<int> indices =
      context.env.robotConfig().resolveJointIndices(
          selectedJointNames(params, context));
  return makeFunctionTerm(
      name, context.env, static_cast<int>(indices.size()),
      [indices](RlEnvModule &, const StepContext &step) {
        Eigen::VectorXd result(static_cast<Eigen::Index>(indices.size()));
        for (std::size_t index = 0; index < indices.size(); ++index)
          result(static_cast<Eigen::Index>(index)) =
              step.status.joint_velocity(indices[index]);
        return result;
      });
}

} // namespace igris_c_inference::policy::observation
