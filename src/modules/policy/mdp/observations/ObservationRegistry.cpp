#include "igris_c_inference/modules/policy/mdp/observations/ObservationTerm.h"

#include <stdexcept>
#include <utility>

namespace igris_c_inference::policy::observation {
namespace {

class FunctionTerm final : public Term {
public:
  FunctionTerm(std::string name, RlEnvModule &env, int size, Function function)
      : Term(std::move(name), env), size_(size),
        function_(std::move(function)) {}

  Eigen::VectorXd compute(const StepContext &context) override {
    return function_(env(), context);
  }
  int size() const override { return size_; }

private:
  int size_{0};
  Function function_;
};

} // namespace

std::unique_ptr<Term> makeFunctionTerm(const std::string &name,
                                       RlEnvModule &env, int size,
                                       Function function) {
  if (size <= 0)
    throw std::runtime_error("observation term '" + name +
                             "' must declare a positive size");
  return std::make_unique<FunctionTerm>(name, env, size, std::move(function));
}

const Registry &registry() {
  static const Registry factories{
      {"command", createCommandObservation},
      {"base_ang_vel", createBaseAngularVelocityObservation},
      {"projected_gravity", createProjectedGravityObservation},
      {"last_actions", createLastActionsObservation},
      {"joint_pos_rel", createRelativeJointPositionObservation},
      {"joint_vel", createJointVelocityObservation},
      {"frame_ang_vel", createFrameAngularVelocityObservation},
      {"frame_projected_gravity", createFrameProjectedGravityObservation},
      {"aligned_joint_pos_rel", createAlignedJointPositionObservation},
      {"aligned_joint_vel", createAlignedJointVelocityObservation},
      {"aligned_last_actions", createAlignedLastActionsObservation},
  };
  return factories;
}

} // namespace igris_c_inference::policy::observation
