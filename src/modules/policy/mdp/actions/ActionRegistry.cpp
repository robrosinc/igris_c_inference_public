#include "igris_c_inference/modules/policy/mdp/actions/ActionTerm.h"

namespace igris_c_inference::policy::action {

Eigen::VectorXd Term::process(const Eigen::VectorXd &actions,
                              const StepContext &) {
  return actions;
}

Eigen::VectorXd Term::resetValue(int size) const {
  return Eigen::VectorXd::Zero(size);
}

const Registry &registry() {
  static const Registry factories = {
      {"joint_position", createJointPositionAction},
      {"safe_joint_position", createSafeJointPositionAction},
      {"motion_command_action", createMotionCommandAction},
  };
  return factories;
}

} // namespace igris_c_inference::policy::action
