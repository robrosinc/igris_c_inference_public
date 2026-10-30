#include "igris_c_inference/robotHandler.h"

#include "igris_c_inference/io/robot_io.hpp"
#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/managers/CommandManager.h"
#include "igris_c_inference/modules/policy/managers/TerminationManager.h"
#include "igris_c_inference/modules/state_handler.hpp"

#include "igris_c_sdk/igris_c_msgs.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <thread>

namespace igris_c_inference {
namespace {

std::atomic<bool> interrupted{false};
void handleSignal(int) { interrupted.store(true, std::memory_order_relaxed); }

void printUsage(const char *program) {
  std::cout << "Usage: " << program
            << " --config PATH [--domain-id N] [--namespace NS]"
               " [--cyclonedds-xml PATH]\n";
}

bool parseInteger(const char *text, int &value) {
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0')
    return false;
  value = static_cast<int>(parsed);
  return true;
}

struct Options {
  std::string configPath;
  bool showHelp{false};
  bool overrideDomain{false};
  bool overrideNamespace{false};
  bool overrideXml{false};
  int domain{0};
  std::string robotNamespace;
  std::string xmlPath;
};

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "-h" || argument == "--help") {
      options.showHelp = true;
    } else if (argument == "--config" && index + 1 < argc) {
      options.configPath = argv[++index];
    } else if (argument == "--domain-id" && index + 1 < argc) {
      options.overrideDomain = parseInteger(argv[++index], options.domain);
      if (!options.overrideDomain)
        throw std::runtime_error("invalid --domain-id");
    } else if (argument == "--namespace" && index + 1 < argc) {
      options.overrideNamespace = true;
      options.robotNamespace = argv[++index];
    } else if (argument == "--cyclonedds-xml" && index + 1 < argc) {
      options.overrideXml = true;
      options.xmlPath = argv[++index];
    } else {
      throw std::runtime_error("unexpected argument: " + argument);
    }
  }
  return options;
}

} // namespace

RobotHandler::RobotHandler(int argc, char **argv) : argc_(argc), argv_(argv) {}
RobotHandler::~RobotHandler() { stop(); }

int RobotHandler::initialize() {
  try {
    const Options options = parseOptions(argc_, argv_);
    if (options.showHelp) {
      printUsage(argv_[0]);
      return 1;
    }
    if (options.configPath.empty()) {
      printUsage(argv_[0]);
      return -1;
    }
    config_ = LoadInferenceConfig(options.configPath);
    if (options.overrideDomain)
      config_.robot.domain_id = options.domain;
    if (options.overrideNamespace)
      config_.robot.robot_namespace = options.robotNamespace;
    if (options.overrideXml)
      config_.robot.cyclonedds_xml_path = options.xmlPath;
    robotConfig_ = LoadRobotConfig(config_.robot_path);

    robotIo_ = std::make_unique<RobotIo>();
    stateHandler_ = std::make_unique<StateHandler>();
    stateHandler_->configure(robotConfig_);
    rlEnv_ = std::make_unique<RlEnvModule>("igris_c_inference", config_,
                                           robotConfig_);
    if (!robotIo_->initialize(config_.robot))
      throw std::runtime_error("failed to initialize robot DDS I/O");
    if (rlEnv_->initialize() != 0)
      throw std::runtime_error("failed to initialize policy environment");
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);
    return 0;
  } catch (const std::exception &exception) {
    transition(RuntimeState::ERROR, exception.what());
    stop();
    return -1;
  }
}

int RobotHandler::execute() {
  if (!rlEnv_ || !robotIo_)
    return -1;
  transition(RuntimeState::WAIT, "waiting for LowState and RobotState");
  int result = -1;
  try {
    result = control();
  } catch (const std::exception &exception) {
    stopInference();
    transition(RuntimeState::ERROR, exception.what());
  }
  stop();
  return result;
}

void RobotHandler::stop() {
  stopInference();
  if (rlEnv_)
    rlEnv_->stop();
}

void RobotHandler::stopInference() {
  inferenceRunning_.store(false, std::memory_order_release);
  if (inferenceThread_.joinable())
    inferenceThread_.join();
}

void RobotHandler::transition(RuntimeState next, const std::string &reason) {
  state_ = next;
  const char *name = "WAIT";
  switch (next) {
  case RuntimeState::WAIT:
    break;
  case RuntimeState::IDLE:
    name = "IDLE";
    break;
  case RuntimeState::RUNNING:
    name = "RUNNING";
    break;
  case RuntimeState::ERROR:
    name = "ERROR";
    break;
  }
  std::cerr << "RobotHandler: \033[31m" << name << "\033[0m: " << reason
            << '\n';
}

bool RobotHandler::readState(
    ProcessedState &state, std::string &error,
    std::chrono::steady_clock::time_point *received_at) const {
  try {
    igris_c_sdk::msg::dds_::LowState_ rawState;
    uint64_t sequence = 0;
    if (!robotIo_->snapshotState(rawState, sequence, received_at)) {
      error = "latest LowState is unavailable";
      return false;
    }
    if (!stateHandler_->update(rawState, sequence, state)) {
      error = "latest LowState is invalid";
      return false;
    }
    error.clear();
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
}

void RobotHandler::inference() {
  using Clock = std::chrono::steady_clock;
  const auto period = rlEnv_->inferencePeriod();
  auto next = Clock::now() + period;
  try {
    while (inferenceRunning_.load(std::memory_order_acquire) &&
           !interrupted.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_until(next);
      if (!inferenceRunning_.load(std::memory_order_acquire))
        break;
      ProcessedState state;
      std::string error;
      Clock::time_point received_at;
      if (!readState(state, error, &received_at) ||
          Clock::now() - received_at >
              std::chrono::milliseconds(config_.state_timeout_ms) ||
          rlEnv_->runInferenceWorkerStep(state) != 0) {
        inferenceFailed_.store(true, std::memory_order_release);
        break;
      }
      next += period;
      const auto now = Clock::now();
      if (next <= now)
        next = now + period;
    }
  } catch (const std::exception &exception) {
    std::cerr << "RobotHandler: inference worker: " << exception.what() << '\n';
    inferenceFailed_.store(true, std::memory_order_release);
  }
}

int RobotHandler::control() {
  using Clock = std::chrono::steady_clock;
  const auto control_period = std::chrono::microseconds(std::max<int64_t>(
      1, static_cast<int64_t>(1000000.0 / config_.control_hz)));
  const auto idle_period = std::chrono::milliseconds(20);
  const auto state_timeout =
      std::chrono::milliseconds(config_.state_timeout_ms);
  bool mode_request_attempted = false;
  Clock::time_point mode_requested_at{};
  bool await_mode_exit = false;
  bool policy_reset_pending = true;
  bool stale_state_error = false;
  uint64_t running_mode_generation = 0;
  auto next_idle_inference = Clock::now();
  auto next = Clock::now();
  while (!interrupted.load(std::memory_order_relaxed)) {
    uint8_t robot_mode = 0;
    uint64_t low_mode_generation = 0;
    Clock::time_point low_mode_entered_at;
    const bool has_mode = robotIo_->snapshotRobotMode(
        robot_mode, &low_mode_generation, &low_mode_entered_at);
    const bool low_mode =
        has_mode &&
        robot_mode ==
            igris_c_sdk::msg::dds_::RobotState_Constants::ROBOT_STATE_LOW;
    if (low_mode)
      mode_requested_at = Clock::time_point{};
    if (await_mode_exit && has_mode && !low_mode)
      await_mode_exit = false;

    ProcessedState state;
    std::string error;
    Clock::time_point received_at;
    const bool has_state = readState(state, error, &received_at);
    const bool fresh_state =
        has_state && Clock::now() - received_at <= state_timeout;
    const bool state_after_mode_entry =
        !low_mode || (has_state && received_at >= low_mode_entered_at);

    if (state_ == RuntimeState::WAIT) {
      if (has_mode && has_state) {
        transition(RuntimeState::IDLE, "initial robot state received");
      } else if (has_mode && error != "latest LowState is unavailable" &&
                 !error.empty()) {
        transition(RuntimeState::ERROR, error);
      }
    } else if (state_ == RuntimeState::IDLE) {
      if (!has_state && error != "latest LowState is unavailable") {
        transition(RuntimeState::ERROR, error);
      } else if (mode_request_attempted && !low_mode &&
                 mode_requested_at != Clock::time_point{} &&
                 Clock::now() - mode_requested_at >
                     std::chrono::milliseconds(
                         config_.low_level_mode_timeout_ms)) {
        transition(RuntimeState::ERROR, "low-level mode was not confirmed");
      } else if (fresh_state && policy_reset_pending) {
        if (rlEnv_->reset(state) != 0)
          transition(RuntimeState::ERROR, "failed to reset policy environment");
        else {
          policy_reset_pending = false;
          next_idle_inference = Clock::now();
        }
      } else if (fresh_state) {
        const bool commands_ready = rlEnv_->commandsReady();
        if (config_.update_while_idle && commands_ready) {
          if (Clock::now() >= next_idle_inference) {
            if (rlEnv_->runInferenceWorkerStep(state) != 0)
              transition(RuntimeState::ERROR, "idle inference failed");
            next_idle_inference += rlEnv_->inferencePeriod();
            if (next_idle_inference <= Clock::now())
              next_idle_inference = Clock::now() + rlEnv_->inferencePeriod();
          }
          if (state_ == RuntimeState::IDLE) {
            InferenceCommand idle_command;
            if (rlEnv_->compute(state, idle_command) != 0)
              transition(RuntimeState::ERROR, "idle control update failed");
          }
        }
        if (state_ == RuntimeState::IDLE && state_after_mode_entry &&
            commands_ready && !await_mode_exit) {
          if (!low_mode && config_.enter_low_level_mode &&
              !mode_request_attempted) {
            mode_request_attempted = true;
            InferenceCommand warmup_command;
            if (rlEnv_->runInferenceWorkerStep(state) != 0 ||
                rlEnv_->compute(state, warmup_command) != 0) {
              transition(RuntimeState::ERROR,
                         "pre-mode inference or command failed");
            } else if (!robotIo_->requestLowLevelMode(
                           std::chrono::milliseconds(
                               config_.low_level_mode_timeout_ms),
                           error)) {
              transition(RuntimeState::ERROR,
                         "failed to enter low-level control mode: " + error);
            } else {
              mode_requested_at = Clock::now();
            }
          } else if (low_mode) {
            InferenceCommand first_command;
            if (config_.update_while_idle)
              rlEnv_->resetHeadingAlignment();
            if (rlEnv_->runInferenceWorkerStep(state) != 0 ||
                rlEnv_->compute(state, first_command) != 0) {
              transition(RuntimeState::ERROR,
                         "initial inference or command failed");
            } else {
              mode_request_attempted = true;
              running_mode_generation = low_mode_generation;
              inferenceFailed_.store(false, std::memory_order_release);
              inferenceRunning_.store(true, std::memory_order_release);
              inferenceThread_ = std::thread(&RobotHandler::inference, this);
              transition(RuntimeState::RUNNING, "low-level mode confirmed");
            }
          }
        }
      }
    } else if (state_ == RuntimeState::RUNNING) {
      if (!low_mode) {
        stopInference();
        policy_reset_pending = true;
        transition(RuntimeState::IDLE, "robot left low-level mode");
      } else if (low_mode_generation != running_mode_generation) {
        stopInference();
        policy_reset_pending = true;
        transition(RuntimeState::IDLE, "low-level mode re-entered");
      } else if (inferenceFailed_.load(std::memory_order_acquire)) {
        stopInference();
        transition(RuntimeState::ERROR, "inference worker failed");
      } else if (!has_state || !fresh_state) {
        stopInference();
        stale_state_error = has_state;
        transition(RuntimeState::ERROR,
                   has_state ? "LowState is stale" : error);
      } else {
        InferenceCommand command;
        if (state_ != RuntimeState::ERROR &&
            rlEnv_->compute(state, command) != 0) {
          stopInference();
          transition(RuntimeState::ERROR, "control command failed");
        } else if (state_ != RuntimeState::ERROR) {
          policy::TerminationRequest request;
          if (rlEnv_->consumeTerminationRequest(request) &&
              request.outcome == policy::TerminationOutcomeType::STOP) {
            stopInference();
            await_mode_exit = true;
            policy_reset_pending = true;
            transition(RuntimeState::IDLE,
                       "termination requested by " + request.termName);
          } else if (!robotIo_->publish(command)) {
            stopInference();
            transition(RuntimeState::ERROR, "failed to publish LowCmd");
          }
        }
      }
    } else if (state_ == RuntimeState::ERROR && stale_state_error &&
               fresh_state) {
      stale_state_error = false;
      inferenceFailed_.store(false, std::memory_order_release);
      policy_reset_pending = true;
      await_mode_exit = low_mode;
      transition(RuntimeState::IDLE, "fresh LowState received");
    }
    next += state_ == RuntimeState::RUNNING ||
                    (state_ == RuntimeState::IDLE && config_.update_while_idle)
                ? control_period
                : idle_period;
    if (next < Clock::now())
      next = Clock::now();
    std::this_thread::sleep_until(next);
  }
  return state_ == RuntimeState::ERROR ? -1 : 0;
}

} // namespace igris_c_inference
