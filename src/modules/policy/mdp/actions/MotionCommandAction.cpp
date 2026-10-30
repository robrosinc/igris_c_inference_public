#include "igris_c_inference/modules/policy/mdp/actions/ActionTerm.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/managers/CommandManager.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace igris_c_inference::policy::action {
namespace {

class MotionCommandAction final : public Term {
public:
  MotionCommandAction(std::string name, RlEnvModule &env,
                      const YAML::Node &config)
      : Term(std::move(name), env) {
    const std::string where = "actions.terms." + this->name();
    const YAML::Node params = yaml::optionalMap(config, "params", where);
    yaml::checkKeys(params, {"command", "offset", "joints", "kp", "kd"},
                    where + ".params");

    command_ =
        yaml::required<std::string>(params, "command", where + ".params");
    if (command_.empty())
      throw std::runtime_error(where + ".params.command must not be empty");

    const int offset = yaml::required<int>(params, "offset", where + ".params");
    if (offset < 0)
      throw std::runtime_error(where + ".params.offset must not be negative");
    offset_ = static_cast<Eigen::Index>(offset);

    const std::vector<std::string> joint_names =
        yaml::readNames(params["joints"], where + ".params.joints");
    joint_indices_ = env.robotConfig().resolveJointIndices(joint_names);
    kp_ = yaml::readVector(params["kp"], joint_names.size(),
                           where + ".params.kp");
    kd_ = yaml::readVector(params["kd"], joint_names.size(),
                           where + ".params.kd");
    if ((kp_.array() < 0.0).any() || (kd_.array() < 0.0).any())
      throw std::runtime_error(where + ".params gains must not be negative");

    const int signal_size = env.commandManager().signalSize(command_);
    const Eigen::Index required_size =
        offset_ + static_cast<Eigen::Index>(joint_indices_.size());
    if (signal_size <= 0)
      throw std::runtime_error(where +
                               ".params.command is not declared: " + command_);
    if (static_cast<Eigen::Index>(signal_size) < required_size)
      throw std::runtime_error(
          where + ".params command slice exceeds signal: " + command_);
  }

  void apply(const Eigen::VectorXd &, const StepContext &context,
             InferenceCommand &command) const override {
    Eigen::VectorXd reference;
    if (!env().commandManager().getSignal(command_, context, reference))
      throw std::runtime_error("motion command signal is unavailable: " +
                               command_);
    const Eigen::Index required_size =
        offset_ + static_cast<Eigen::Index>(joint_indices_.size());
    if (reference.size() < required_size)
      throw std::runtime_error("motion command signal changed size: " +
                               command_);

    const Eigen::VectorXd targets = reference.segment(
        offset_, static_cast<Eigen::Index>(joint_indices_.size()));
    if (!targets.allFinite())
      throw std::runtime_error("motion command targets are non-finite: " +
                               command_);

    for (std::size_t index = 0; index < joint_indices_.size(); ++index) {
      const int joint = joint_indices_[index];
      const Eigen::Index value_index = static_cast<Eigen::Index>(index);
      command.q(joint) = targets(value_index);
      command.dq(joint) = 0.0;
      command.tau(joint) = 0.0;
      command.kp(joint) = kp_(value_index);
      command.kd(joint) = kd_(value_index);
    }
  }

  int actionSize() const override { return 0; }
  bool consumesPolicyOutput() const override { return false; }

private:
  std::string command_;
  Eigen::Index offset_{0};
  std::vector<int> joint_indices_;
  Eigen::VectorXd kp_;
  Eigen::VectorXd kd_;
};

} // namespace

std::unique_ptr<Term> createMotionCommandAction(const std::string &name,
                                                const YAML::Node &config,
                                                const FactoryContext &context) {
  const std::string where = "actions.terms." + name;
  yaml::checkKeys(config, {"function", "params"}, where);
  return std::make_unique<MotionCommandAction>(name, context.env, config);
}

} // namespace igris_c_inference::policy::action
