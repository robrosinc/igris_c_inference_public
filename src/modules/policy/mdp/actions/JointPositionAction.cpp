#include "igris_c_inference/modules/policy/mdp/actions/JointActionTerm.h"

#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <memory>
#include <utility>

namespace igris_c_inference::policy::action {
namespace {

class JointPositionAction final : public JointActionTerm {
public:
  JointPositionAction(std::string name, RlEnvModule &env,
                      const YAML::Node &config)
      : JointActionTerm(std::move(name), env, config) {}

  void apply(const Eigen::VectorXd &actions, const StepContext &,
             InferenceCommand &command) const override {
    for (std::size_t index = 0; index < jointIndices().size(); ++index) {
      const int joint = jointIndices()[index];
      const Eigen::Index action_index = static_cast<Eigen::Index>(index);
      command.q(joint) = defaultPositions()(action_index) +
                         actions(action_index) * actionScale()(action_index);
      command.dq(joint) = 0.0;
      command.tau(joint) = 0.0;
      command.kp(joint) = kp()(action_index);
      command.kd(joint) = kd()(action_index);
    }
    applyKinematicModes(command);
  }
};

} // namespace

std::unique_ptr<Term> createJointPositionAction(const std::string &name,
                                                const YAML::Node &config,
                                                const FactoryContext &context) {
  const std::string where = "actions.terms." + name;
  yaml::checkKeys(config,
                  {"function", "output", "params", "action_joints",
                   "action_scale", "kp", "kd", "output_clip", "lpf_cutoff_hz"},
                  where);
  yaml::checkKeys(yaml::optionalMap(config, "params", where), {},
                  where + ".params");
  return std::make_unique<JointPositionAction>(name, context.env, config);
}

} // namespace igris_c_inference::policy::action
