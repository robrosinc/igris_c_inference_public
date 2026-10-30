#pragma once

#include "igris_c_inference/core/config.hpp"
#include "igris_c_inference/core/pipeline_types.hpp"
#include "igris_c_inference/core/types.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

namespace igris_c_inference {

class RobotIo;
class RlEnvModule;
class StateHandler;

class RobotHandler {
public:
  RobotHandler(int argc, char **argv);
  ~RobotHandler();
  RobotHandler(const RobotHandler &) = delete;
  RobotHandler &operator=(const RobotHandler &) = delete;

  int initialize();
  int execute();
  void stop();

private:
  enum class RuntimeState { WAIT, IDLE, RUNNING, ERROR };

  bool
  readState(ProcessedState &state, std::string &error,
            std::chrono::steady_clock::time_point *received_at = nullptr) const;
  void transition(RuntimeState next, const std::string &reason);
  void stopInference();
  void inference();
  int control();

  int argc_{0};
  char **argv_{nullptr};
  InferenceConfig config_;
  RobotConfig robotConfig_;
  std::unique_ptr<RobotIo> robotIo_;
  std::unique_ptr<StateHandler> stateHandler_;
  std::unique_ptr<RlEnvModule> rlEnv_;
  RuntimeState state_{RuntimeState::WAIT};
  std::atomic<bool> inferenceRunning_{false};
  std::atomic<bool> inferenceFailed_{false};
  std::thread inferenceThread_;
};

} // namespace igris_c_inference
