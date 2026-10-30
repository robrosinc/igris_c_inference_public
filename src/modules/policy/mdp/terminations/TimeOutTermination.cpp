#include "igris_c_inference/modules/policy/mdp/terminations/TerminationTerm.h"

#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace igris_c_inference::policy::termination {
namespace {

class TimeOutTermination final : public Term {
public:
  TimeOutTermination(std::string name, RlEnvModule &env, double durationSeconds)
      : Term(std::move(name), env), durationSeconds_(durationSeconds) {}

  bool compute(const StepContext &context) override {
    if (!std::isfinite(context.dt) || context.dt <= 0.0)
      throw std::runtime_error(
          "time_out requires a positive finite step duration");
    const uint64_t step = context.phase == ExecutionPhase::CONTROL
                              ? context.controlStep
                              : context.policyStep;
    return (static_cast<double>(step) + 1.0) * context.dt >= durationSeconds_;
  }

private:
  double durationSeconds_{0.0};
};

} // namespace

std::unique_ptr<Term> createTimeOutTermination(const std::string &name,
                                               const YAML::Node &params,
                                               const FactoryContext &context) {
  yaml::checkKeys(params, {"duration_s"}, "termination.term.params");
  const double durationSeconds =
      yaml::required<double>(params, "duration_s", "termination.term.params");
  if (!std::isfinite(durationSeconds) || durationSeconds <= 0.0)
    throw std::runtime_error(
        "termination.term.params.duration_s must be positive and finite");
  return std::make_unique<TimeOutTermination>(name, context.env,
                                              durationSeconds);
}

} // namespace igris_c_inference::policy::termination
