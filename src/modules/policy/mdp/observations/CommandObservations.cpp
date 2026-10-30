#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/managers/CommandManager.h"
#include "igris_c_inference/modules/policy/mdp/observations/ObservationTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <stdexcept>
#include <utility>

namespace igris_c_inference::policy::observation {
namespace {

PolicySignal lookup(RlEnvModule &env, const std::string &name,
                    const StepContext &context) {
  PolicySignal signal;
  if (!env.commandManager().getSignal(name, context, signal))
    throw std::runtime_error("missing observation command: " + name);
  return signal;
}

} // namespace

std::unique_ptr<Term> createCommandObservation(const std::string &name,
                                               const YAML::Node &params,
                                               const FactoryContext &context) {
  yaml::checkKeys(params, {"name", "offset", "size"},
                  "observation.term.params");
  const std::string commandName =
      yaml::required<std::string>(params, "name", "observation.term.params");
  const bool hasOffset = static_cast<bool>(params["offset"]);
  const bool hasSize = static_cast<bool>(params["size"]);
  if (hasOffset != hasSize)
    throw std::runtime_error(
        "command observation requires both offset and size when slicing");
  const int signalSize = context.env.commandManager().signalSize(commandName);
  if (signalSize <= 0)
    throw std::runtime_error("observation command is not declared: " +
                             commandName);
  const int offset = hasOffset ? params["offset"].as<int>() : 0;
  const int size = hasSize ? params["size"].as<int>() : signalSize;
  if (offset < 0 || size <= 0 || offset > signalSize ||
      size > signalSize - offset)
    throw std::runtime_error("command observation slice exceeds signal: " +
                             commandName);
  return makeFunctionTerm(
      name, context.env, size,
      [commandName, offset, size](RlEnvModule &env, const StepContext &step) {
        const PolicySignal signal = lookup(env, commandName, step);
        if (signal.value.size() < offset + size)
          throw std::runtime_error("command observation size changed: " +
                                   commandName);
        return Eigen::VectorXd(signal.value.segment(offset, size));
      });
}

} // namespace igris_c_inference::policy::observation
