#include "igris_c_inference/modules/policy/mdp/actions/JointActionTerm.h"

#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace igris_c_inference::policy::action {
namespace {

class SafeJointPositionAction final : public JointActionTerm {
public:
  SafeJointPositionAction(std::string name, RlEnvModule &env,
                          const YAML::Node &config)
      : JointActionTerm(std::move(name), env, config) {
    const std::string where = "actions.terms." + this->name();
    const YAML::Node params = yaml::optionalMap(config, "params", where);
    yaml::checkKeys(params,
                    {"unselected_joint_behavior", "position_limit_recovery"},
                    where + ".params");
    unselected_behavior_ =
        params["unselected_joint_behavior"]
            ? params["unselected_joint_behavior"].as<std::string>()
            : "default";
    if (unselected_behavior_ != "default" &&
        unselected_behavior_ != "hold_current")
      throw std::runtime_error(
          where +
          ".params.unselected_joint_behavior must be default or hold_current");
    if (params["position_limit_recovery"])
      configureRecovery(params["position_limit_recovery"], where);
  }

  void apply(const Eigen::VectorXd &actions, const StepContext &context,
             InferenceCommand &command) const override {
    if (!context.status.valid)
      throw std::runtime_error("current robot state is invalid");
    if (unselected_behavior_ == "hold_current")
      command.q = context.status.joint_position;

    for (std::size_t index = 0; index < jointIndices().size(); ++index) {
      const int joint = jointIndices()[index];
      const Eigen::Index action_index = static_cast<Eigen::Index>(index);
      double target = defaultPositions()(action_index) +
                      actions(action_index) * actionScale()(action_index);
      if (recovery_ratio_ > 0.0 &&
          recovery_exclusions_.count(jointNames()[index]) == 0U) {
        const double low = positionMin()(action_index);
        const double high = positionMax()(action_index);
        const double midpoint = 0.5 * (low + high);
        const double soft_low = midpoint - 0.5 * (high - low) * recovery_ratio_;
        const double soft_high =
            midpoint + 0.5 * (high - low) * recovery_ratio_;
        const double current = context.status.joint_position(joint);
        if (current > soft_high && target > high) {
          target -= std::clamp((current - soft_high) /
                                   std::max(1.0e-6, high - soft_high),
                               0.0, 1.0) *
                    (target - high);
        } else if (current < soft_low && target < low) {
          target += std::clamp((soft_low - current) /
                                   std::max(1.0e-6, soft_low - low),
                               0.0, 1.0) *
                    (low - target);
        }
      }
      command.q(joint) = target;
      command.dq(joint) = 0.0;
      command.tau(joint) = 0.0;
      command.kp(joint) = kp()(action_index);
      command.kd(joint) = kd()(action_index);
    }
    applyKinematicModes(command);
  }

private:
  void configureRecovery(const YAML::Node &recovery, const std::string &where) {
    yaml::checkKeys(recovery, {"soft_limit_ratio", "excluded_joints"},
                    where + ".params.position_limit_recovery");
    recovery_ratio_ =
        yaml::required<double>(recovery, "soft_limit_ratio",
                               where + ".params.position_limit_recovery");
    if (!std::isfinite(recovery_ratio_) || recovery_ratio_ <= 0.0 ||
        recovery_ratio_ > 1.0)
      throw std::runtime_error(
          where +
          ".params.position_limit_recovery.soft_limit_ratio must be in (0, 1]");
    if (!recovery["excluded_joints"])
      return;
    const auto names = yaml::readNames(
        recovery["excluded_joints"],
        where + ".params.position_limit_recovery.excluded_joints", true);
    for (const std::string &name : names) {
      robot().joint(name);
      recovery_exclusions_.insert(name);
    }
  }

  std::string unselected_behavior_;
  double recovery_ratio_ = -1.0;
  std::unordered_set<std::string> recovery_exclusions_;
};

} // namespace

std::unique_ptr<Term>
createSafeJointPositionAction(const std::string &name, const YAML::Node &config,
                              const FactoryContext &context) {
  const std::string where = "actions.terms." + name;
  yaml::checkKeys(config,
                  {"function", "output", "params", "action_joints",
                   "action_scale", "kp", "kd", "output_clip", "lpf_cutoff_hz"},
                  where);
  return std::make_unique<SafeJointPositionAction>(name, context.env, config);
}

} // namespace igris_c_inference::policy::action
