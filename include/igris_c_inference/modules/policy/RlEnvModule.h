#pragma once

#include "igris_c_inference/core/config.hpp"
#include "igris_c_inference/modules/policy/runtime/AsyncPolicyModule.h"
#include "igris_c_inference/modules/policy/runtime/RlEnvConfig.h"
#include "igris_c_inference/modules/policy/utils/PolicyPipelineTypes.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace igris_c_inference::policy {
class ActionManager;
class CommandManager;
class EventManager;
class ObservationManager;
class PolicyRuntime;
class TerminationManager;
struct TerminationRequest;
enum class ObservationUpdatePhase;
} // namespace igris_c_inference::policy

namespace igris_c_inference {

class RlEnvModule : public AsyncPolicyModule {
public:
  RlEnvModule(std::string policyLabel, const InferenceConfig &applicationConfig,
              const RobotConfig &robotConfig);
  ~RlEnvModule() override;

  RlEnvModule(const RlEnvModule &) = delete;
  RlEnvModule &operator=(const RlEnvModule &) = delete;

  int initialize();
  int reset(const ProcessedState &status);
  int runInferenceWorkerStep(const ProcessedState &status) override;
  int compute(const ProcessedState &status, InferenceCommand &controlCommand);
  std::chrono::microseconds inferencePeriod() const override;
  void stop();
  bool waitUntilReady();
  bool commandsReady();
  void resetHeadingAlignment();

  policy::ActionManager &actionManager();
  const policy::ActionManager &actionManager() const;
  policy::CommandManager &commandManager();
  const policy::CommandManager &commandManager() const;
  policy::ObservationManager &observationManager();
  const policy::ObservationManager &observationManager() const;
  policy::TerminationManager &terminationManager();
  const policy::TerminationManager &terminationManager() const;
  policy::EventManager &eventManager();
  const policy::EventManager &eventManager() const;
  bool consumeTerminationRequest(policy::TerminationRequest &request);

  const std::string &label() const { return policyLabel_; }
  const InferenceConfig &applicationConfig() const {
    return applicationConfig_;
  }
  const RobotConfig &robotConfig() const { return robotConfig_; }
  const policy::RlEnvConfig &envConfig() const { return envConfig_; }

private:
  int configurePolicy(const std::string &modelPath,
                      const std::string &policyConfigPath);
  int configurePolicyPipeline(const std::string &modelPath,
                              const std::string &policyConfigPath);
  int initializePolicyObservations(const policy::StepContext &context,
                                   std::string &error);
  int runPolicyStep(const ProcessedState &status, uint64_t epoch,
                    std::string &error);
  int applyLatestPolicyAction(const policy::StepContext &context,
                              InferenceCommand &controlCommand,
                              std::string &error);
  int resetPolicyExecution(const ProcessedState *status, bool applyResetEvents,
                           std::string &error);
  int runPolicy(const policy::TensorMap &inputs, policy::TensorMap &outputs,
                std::string &error);
  int updatePolicyObservations(const policy::StepContext &context,
                               policy::ObservationUpdatePhase phase,
                               std::string &error);
  int preparePolicyCommands(const policy::StepContext &context,
                            std::string &error);
  int commitPolicyCommands(const policy::StepContext &context,
                           const policy::TensorMap &outputs,
                           const policy::ActionFrame &actions,
                           std::string &error);
  int buildPolicyInputs(policy::TensorMap &inputs, std::string &error);
  int processPolicyActions(const policy::StepContext &context,
                           const policy::TensorMap &outputs,
                           policy::ActionFrame &processedActions,
                           std::string &error);
  int buildPolicyControlCommand(const policy::ActionFrame &actions,
                                const policy::StepContext &context,
                                InferenceCommand &controlCommand,
                                std::string &error);
  int resetPolicy(const policy::LifecycleContext &context,
                  bool applyResetEvents, std::string &error);
  int validatePolicyContracts(std::string &error) const;
  void logPolicySummary() const;
  void reportPolicyFailure(const std::string &stage, const std::string &detail);
  void reportPolicyRecovery(const std::string &stage);

  struct DiagnosticState {
    std::string error;
    uint64_t failureCount{0};
  };

  std::string policyLabel_;
  InferenceConfig applicationConfig_;
  RobotConfig robotConfig_;
  policy::RlEnvConfig envConfig_;
  std::unique_ptr<policy::PolicyRuntime> policyRuntime_;
  std::unique_ptr<policy::CommandManager> commandManager_;
  std::unique_ptr<policy::ObservationManager> observationManager_;
  std::unique_ptr<policy::ActionManager> actionManager_;
  std::unique_ptr<policy::TerminationManager> terminationManager_;
  std::unique_ptr<policy::EventManager> eventManager_;
  std::chrono::microseconds inferencePeriod_{20000};
  policy::ActionFrame activePolicyActions_;
  uint64_t lastConsumedPolicySequence_{0};
  std::atomic<bool> active_{false};
  std::atomic<uint64_t> executionEpoch_{0};
  std::atomic<uint64_t> readyEpoch_{0};
  std::atomic<uint64_t> controlStep_{0};
  std::mutex inferenceStepMutex_;
  std::mutex stateMutationMutex_;
  std::mutex diagnosticMutex_;
  std::unordered_map<std::string, DiagnosticState> diagnostics_;
};

} // namespace igris_c_inference
