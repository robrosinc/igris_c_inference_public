#include "igris_c_inference/modules/policy/RlEnvModule.h"

#include "igris_c_inference/modules/policy/managers/ActionManager.h"
#include "igris_c_inference/modules/policy/managers/CommandManager.h"
#include "igris_c_inference/modules/policy/managers/EventManager.h"
#include "igris_c_inference/modules/policy/managers/ObservationManager.h"
#include "igris_c_inference/modules/policy/managers/TerminationManager.h"
#include "igris_c_inference/modules/policy/runtime/PolicyRuntime.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace igris_c_inference {
namespace {

std::string descriptorText(const policy::TermDescriptor &descriptor) {
  std::ostringstream stream;
  stream << descriptor.name << '=' << descriptor.function;
  if (descriptor.size > 0)
    stream << '[' << descriptor.size << ']';
  if (!descriptor.binding.empty())
    stream << " -> " << descriptor.binding;
  if (!descriptor.updatePhase.empty())
    stream << " @" << descriptor.updatePhase;
  return stream.str();
}

} // namespace

RlEnvModule::RlEnvModule(std::string policyLabel,
                         const InferenceConfig &applicationConfig,
                         const RobotConfig &robotConfig)
    : policyLabel_(std::move(policyLabel)),
      applicationConfig_(applicationConfig), robotConfig_(robotConfig),
      policyRuntime_(std::make_unique<policy::PolicyRuntime>(policyLabel_)),
      commandManager_(std::make_unique<policy::CommandManager>(*this)),
      observationManager_(std::make_unique<policy::ObservationManager>(*this)),
      actionManager_(std::make_unique<policy::ActionManager>(*this)),
      terminationManager_(std::make_unique<policy::TerminationManager>(*this)),
      eventManager_(std::make_unique<policy::EventManager>(*this)) {}

RlEnvModule::~RlEnvModule() { stop(); }

int RlEnvModule::initialize() {
  if (configurePolicyPipeline(applicationConfig_.policy_path,
                              applicationConfig_.policy_config_path) != 0)
    return -1;
  std::string error;
  if (!commandManager_->initialize(error)) {
    reportPolicyFailure("command initialization", error);
    return -1;
  }
  return 0;
}

int RlEnvModule::reset(const ProcessedState &status) {
  active_.store(false, std::memory_order_release);
  std::string error;
  if (resetPolicyExecution(&status, true, error) != 0) {
    reportPolicyFailure("reset", error);
    return -1;
  }
  active_.store(true, std::memory_order_release);
  reportPolicyRecovery("reset");
  return 0;
}

int RlEnvModule::runInferenceWorkerStep(const ProcessedState &status) {
  if (!active_.load(std::memory_order_acquire))
    return 0;
  std::lock_guard<std::mutex> inferenceLock(inferenceStepMutex_);
  if (!active_.load(std::memory_order_acquire))
    return 0;
  const uint64_t epoch = executionEpoch_.load(std::memory_order_acquire);
  if (epoch == 0U || readyEpoch_.load(std::memory_order_acquire) != epoch)
    return 0;
  std::string error;
  if (runPolicyStep(status, epoch, error) != 0) {
    reportPolicyFailure("inference", error);
    return -1;
  }
  reportPolicyRecovery("inference");
  return 0;
}

int RlEnvModule::compute(const ProcessedState &status,
                         InferenceCommand &controlCommand) {
  const uint64_t epoch = executionEpoch_.load(std::memory_order_acquire);
  if (!active_.load(std::memory_order_acquire) ||
      readyEpoch_.load(std::memory_order_acquire) != epoch)
    return -1;
  std::lock_guard<std::mutex> mutationLock(stateMutationMutex_);
  if (!active_.load(std::memory_order_acquire) ||
      executionEpoch_.load(std::memory_order_acquire) != epoch ||
      readyEpoch_.load(std::memory_order_acquire) != epoch)
    return -1;
  const uint64_t step = controlStep_.fetch_add(1U, std::memory_order_relaxed);
  const policy::StepContext context{
      status,
      policy::ExecutionPhase::CONTROL,
      1.0 / static_cast<double>(applicationConfig_.control_hz),
      step,
      commandManager_->policyStep(),
      std::chrono::steady_clock::now()};
  std::string error;
  const int result = applyLatestPolicyAction(context, controlCommand, error);
  if (result != 0)
    reportPolicyFailure("control", error);
  else
    reportPolicyRecovery("control");
  return result;
}

std::chrono::microseconds RlEnvModule::inferencePeriod() const {
  return inferencePeriod_;
}

void RlEnvModule::stop() {
  active_.store(false, std::memory_order_release);
  commandManager_->stop();
}

bool RlEnvModule::waitUntilReady() {
  std::string error;
  const bool ready = commandManager_->waitUntilReady(
      std::chrono::milliseconds(applicationConfig_.command_timeout_ms), error);
  if (!ready)
    reportPolicyFailure("command readiness", error);
  return ready;
}

bool RlEnvModule::commandsReady() {
  std::string error;
  return commandManager_->waitUntilReady(std::chrono::milliseconds(0), error);
}

void RlEnvModule::resetHeadingAlignment() {
  commandManager_->resetHeadingAlignment();
}

policy::ActionManager &RlEnvModule::actionManager() { return *actionManager_; }
const policy::ActionManager &RlEnvModule::actionManager() const {
  return *actionManager_;
}
policy::CommandManager &RlEnvModule::commandManager() {
  return *commandManager_;
}
const policy::CommandManager &RlEnvModule::commandManager() const {
  return *commandManager_;
}
policy::ObservationManager &RlEnvModule::observationManager() {
  return *observationManager_;
}
const policy::ObservationManager &RlEnvModule::observationManager() const {
  return *observationManager_;
}
policy::TerminationManager &RlEnvModule::terminationManager() {
  return *terminationManager_;
}
const policy::TerminationManager &RlEnvModule::terminationManager() const {
  return *terminationManager_;
}
policy::EventManager &RlEnvModule::eventManager() { return *eventManager_; }
const policy::EventManager &RlEnvModule::eventManager() const {
  return *eventManager_;
}
bool RlEnvModule::consumeTerminationRequest(
    policy::TerminationRequest &request) {
  return terminationManager_->consumeRequest(request);
}

int RlEnvModule::configurePolicy(const std::string &modelPath,
                                 const std::string &policyConfigPath) {
  std::string error;
  if (policy::loadRlEnvConfig(policyConfigPath, envConfig_, error) != 0) {
    reportPolicyFailure("policy configuration", error);
    return -1;
  }
  if (policyRuntime_->configure(modelPath, envConfig_, error) != 0) {
    reportPolicyFailure("model configuration", error);
    return -1;
  }
  inferencePeriod_ = std::chrono::microseconds(1000000 / envConfig_.rateHz);
  return 0;
}

int RlEnvModule::configurePolicyPipeline(const std::string &modelPath,
                                         const std::string &policyConfigPath) {
  if (configurePolicy(modelPath, policyConfigPath) != 0)
    return -1;
  std::string error;
  const std::unordered_map<std::string, std::string> resourcePaths;
  if (commandManager_->configure(envConfig_.commandPath, envConfig_.rateHz,
                                 modelPath, resourcePaths, error) != 0 ||
      actionManager_->configure(envConfig_.actionPath, envConfig_.rateHz,
                                error) != 0 ||
      observationManager_->configure(envConfig_.observationPath,
                                     applicationConfig_.control_hz,
                                     envConfig_.rateHz, error) != 0 ||
      terminationManager_->configure(envConfig_.terminationPath,
                                     applicationConfig_.control_hz,
                                     envConfig_.rateHz, error) != 0 ||
      eventManager_->configure(envConfig_.eventPath, error) != 0) {
    reportPolicyFailure("manager configuration", error);
    return -1;
  }
  for (const auto &[outputName, expectedSize] : actionManager_->outputSizes()) {
    const auto output =
        std::find_if(envConfig_.outputs.begin(), envConfig_.outputs.end(),
                     [&outputName](const policy::OutputBinding &binding) {
                       return binding.name == outputName;
                     });
    if (output == envConfig_.outputs.end() ||
        output->size != static_cast<std::size_t>(expectedSize)) {
      reportPolicyFailure("policy contract",
                          "action output binding mismatch: " + outputName);
      return -1;
    }
  }
  if (validatePolicyContracts(error) != 0 ||
      resetPolicyExecution(nullptr, false, error) != 0) {
    reportPolicyFailure("policy contract", error);
    return -1;
  }
  const policy::LifecycleContext startup{policy::LifecyclePhase::STARTUP,
                                         nullptr};
  if (eventManager_->apply("startup", startup, error) != 0) {
    reportPolicyFailure("startup event", error);
    return -1;
  }
  logPolicySummary();
  return 0;
}

int RlEnvModule::initializePolicyObservations(
    const policy::StepContext &context, std::string &error) {
  if (updatePolicyObservations(context, policy::ObservationUpdatePhase::CONTROL,
                               error) != 0)
    return -1;
  policy::StepContext policyContext = context;
  policyContext.phase = policy::ExecutionPhase::POLICY;
  policyContext.dt = 1.0 / static_cast<double>(envConfig_.rateHz);
  return updatePolicyObservations(
      policyContext, policy::ObservationUpdatePhase::POLICY, error);
}

int RlEnvModule::runPolicyStep(const ProcessedState &status, uint64_t epoch,
                               std::string &error) {
  const policy::StepContext context{
      status,
      policy::ExecutionPhase::POLICY,
      1.0 / static_cast<double>(envConfig_.rateHz),
      controlStep_.load(std::memory_order_relaxed),
      commandManager_->policyStep(),
      std::chrono::steady_clock::now()};
  policy::TensorMap inputs;
  policy::TensorMap outputs;
  policy::ActionFrame processedActions;
  if (preparePolicyCommands(context, error) != 0)
    return -1;
  {
    std::lock_guard<std::mutex> mutationLock(stateMutationMutex_);
    if (epoch != executionEpoch_.load(std::memory_order_acquire) ||
        readyEpoch_.load(std::memory_order_acquire) != epoch ||
        !active_.load(std::memory_order_acquire))
      return 0;
    if (updatePolicyObservations(
            context, policy::ObservationUpdatePhase::POLICY, error) != 0 ||
        buildPolicyInputs(inputs, error) != 0)
      return -1;
  }
  if (runPolicy(inputs, outputs, error) != 0)
    return -1;

  std::lock_guard<std::mutex> mutationLock(stateMutationMutex_);
  if (epoch != executionEpoch_.load(std::memory_order_acquire) ||
      readyEpoch_.load(std::memory_order_acquire) != epoch ||
      !active_.load(std::memory_order_acquire))
    return 0;
  if (processPolicyActions(context, outputs, processedActions, error) != 0 ||
      commitPolicyCommands(context, outputs, processedActions, error) != 0)
    return -1;
  if (terminationManager_->update(
          context, policy::TerminationUpdatePhase::POLICY, error) != 0) {
    error = "policy termination update failed: " + error;
    return -1;
  }
  publishInferenceResult(processedActions);
  return 0;
}

int RlEnvModule::applyLatestPolicyAction(const policy::StepContext &context,
                                         InferenceCommand &controlCommand,
                                         std::string &error) {
  if (updatePolicyObservations(context, policy::ObservationUpdatePhase::CONTROL,
                               error) != 0)
    return -1;
  if (terminationManager_->update(
          context, policy::TerminationUpdatePhase::CONTROL, error) != 0) {
    error = "termination update failed: " + error;
    return -1;
  }
  policy::ActionFrame latest;
  if (consumeLatestInferenceResult(lastConsumedPolicySequence_, latest))
    activePolicyActions_ = std::move(latest);
  return buildPolicyControlCommand(activePolicyActions_, context,
                                   controlCommand, error);
}

int RlEnvModule::resetPolicyExecution(const ProcessedState *status,
                                      bool applyResetEvents,
                                      std::string &error) {
  const uint64_t epoch =
      executionEpoch_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
  readyEpoch_.store(0U, std::memory_order_release);
  std::lock_guard<std::mutex> mutationLock(stateMutationMutex_);
  const policy::LifecycleContext lifecycle{policy::LifecyclePhase::RESET,
                                           status};
  if (resetPolicy(lifecycle, applyResetEvents, error) != 0)
    return -1;
  activePolicyActions_ = actionManager_->actionFrame();
  lastConsumedPolicySequence_ = 0;
  controlStep_.store(0U, std::memory_order_relaxed);
  if (status != nullptr) {
    const policy::StepContext initial{
        *status,
        policy::ExecutionPhase::CONTROL,
        1.0 / static_cast<double>(applicationConfig_.control_hz),
        0U,
        0U,
        std::chrono::steady_clock::now()};
    if (initializePolicyObservations(initial, error) != 0)
      return -1;
    readyEpoch_.store(epoch, std::memory_order_release);
  }
  return 0;
}

int RlEnvModule::runPolicy(const policy::TensorMap &inputs,
                           policy::TensorMap &outputs, std::string &error) {
  if (policyRuntime_->run(inputs, outputs, error) != 0) {
    error = "ONNX inference failed: " + error;
    return -1;
  }
  return 0;
}

int RlEnvModule::updatePolicyObservations(const policy::StepContext &context,
                                          policy::ObservationUpdatePhase phase,
                                          std::string &error) {
  if (observationManager_->update(context, phase, error) != 0) {
    error = "observation update failed: " + error;
    return -1;
  }
  return 0;
}

int RlEnvModule::preparePolicyCommands(const policy::StepContext &context,
                                       std::string &error) {
  if (commandManager_->prepare(context, error) != 0) {
    error = "command preparation failed: " + error;
    return -1;
  }
  return 0;
}

int RlEnvModule::commitPolicyCommands(const policy::StepContext &context,
                                      const policy::TensorMap &outputs,
                                      const policy::ActionFrame &actions,
                                      std::string &error) {
  try {
    commandManager_->commit(context, outputs, actions);
    return 0;
  } catch (const std::exception &exception) {
    error = "command commit failed: " + std::string(exception.what());
    return -1;
  }
}

int RlEnvModule::buildPolicyInputs(policy::TensorMap &inputs,
                                   std::string &error) {
  if (observationManager_->buildLatest(inputs, error) != 0) {
    error = "observation build failed: " + error;
    return -1;
  }
  return 0;
}

int RlEnvModule::processPolicyActions(const policy::StepContext &context,
                                      const policy::TensorMap &outputs,
                                      policy::ActionFrame &processedActions,
                                      std::string &error) {
  if (actionManager_->process(outputs, context, processedActions, error) != 0) {
    error = "action processing failed: " + error;
    return -1;
  }
  return 0;
}

int RlEnvModule::buildPolicyControlCommand(const policy::ActionFrame &actions,
                                           const policy::StepContext &context,
                                           InferenceCommand &controlCommand,
                                           std::string &error) {
  if (actionManager_->build(actions, context, controlCommand, error) != 0) {
    error = "action build failed: " + error;
    return -1;
  }
  return 0;
}

int RlEnvModule::resetPolicy(const policy::LifecycleContext &context,
                             bool applyResetEvents, std::string &error) {
  try {
    {
      std::lock_guard<std::mutex> lock(diagnosticMutex_);
      diagnostics_.clear();
    }
    policyRuntime_->reset();
    observationManager_->reset();
    actionManager_->reset();
    commandManager_->reset();
    terminationManager_->reset();
    eventManager_->reset();
    resetInferenceResult();
    if (applyResetEvents &&
        eventManager_->apply("reset", context, error) != 0) {
      error = "event reset failed: " + error;
      return -1;
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = "manager reset failed: " + std::string(exception.what());
    return -1;
  }
}

int RlEnvModule::validatePolicyContracts(std::string &error) const {
  const auto groups = observationManager_->groupSizes();
  const std::unordered_map<std::string, int> groupSizes(groups.begin(),
                                                        groups.end());
  const std::unordered_set<std::string> configuredInputs(
      envConfig_.inputNames.begin(), envConfig_.inputNames.end());
  for (const auto &[name, size] : groups) {
    if (configuredInputs.count(name) == 0U || size <= 0) {
      error = "observation group does not match policy input: " + name;
      return -1;
    }
  }
  for (const std::string &name : envConfig_.inputNames) {
    if (groupSizes.count(name) == 0U) {
      error = "policy input has no observation group: " + name;
      return -1;
    }
  }
  for (const auto &[name, expectedSize] :
       policyRuntime_->expectedInputSizes()) {
    const auto found = groupSizes.find(name);
    if (found != groupSizes.end() &&
        static_cast<std::size_t>(found->second) != expectedSize) {
      error = "observation group size does not match ONNX input: " + name;
      return -1;
    }
  }
  error.clear();
  return 0;
}

void RlEnvModule::logPolicySummary() const {
  const auto logManager = [this](const std::string &name,
                                 const policy::ManagerBase &manager) {
    std::ostringstream stream;
    stream << policyLabel_ << ": " << name << " terms";
    const auto descriptors = manager.termDescriptors();
    if (descriptors.empty()) {
      stream << "=[]";
    } else {
      stream << ": ";
      for (std::size_t index = 0; index < descriptors.size(); ++index) {
        if (index != 0U)
          stream << "; ";
        stream << descriptorText(descriptors[index]);
      }
    }
    std::cout << stream.str() << '\n';
  };
  logManager("command", *commandManager_);
  logManager("observation", *observationManager_);
  logManager("action", *actionManager_);
  logManager("termination", *terminationManager_);
  logManager("event", *eventManager_);
}

void RlEnvModule::reportPolicyFailure(const std::string &stage,
                                      const std::string &detail) {
  const std::string resolved =
      detail.empty() ? "failure returned without diagnostic detail" : detail;
  std::lock_guard<std::mutex> lock(diagnosticMutex_);
  DiagnosticState &state = diagnostics_[stage];
  ++state.failureCount;
  if (state.error == resolved)
    return;
  state.error = resolved;
  std::cerr << policyLabel_ << ": stage=" << stage << " failed: " << resolved
            << '\n';
}

void RlEnvModule::reportPolicyRecovery(const std::string &stage) {
  std::lock_guard<std::mutex> lock(diagnosticMutex_);
  const auto found = diagnostics_.find(stage);
  if (found == diagnostics_.end() || found->second.error.empty())
    return;
  std::cout << policyLabel_ << ": stage=" << stage << " recovered after "
            << found->second.failureCount << " failed step(s)\n";
  diagnostics_.erase(found);
}

} // namespace igris_c_inference
