#include "igris_c_inference/io/robot_io.hpp"

#include "igris_c_sdk/channel_factory.hpp"
#include "igris_c_sdk/qos.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>

namespace igris_c_inference {
namespace {

std_msgs::msg::dds_::Header_ BuildHeader() {
  std_msgs::msg::dds_::Header_ header;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now);
  header.stamp().sec(static_cast<int32_t>(seconds.count()));
  header.stamp().nanosec(static_cast<uint32_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - seconds)
          .count()));
  header.frame_id("igris_c_inference_public");
  return header;
}

} // namespace

bool RobotIo::initialize(const RobotNetworkConfig &config) {
  auto *factory = igris_c_sdk::ChannelFactory::Instance();
  factory->Init(config.domain_id, config.robot_namespace,
                loadCycloneConfig(config.cyclonedds_xml_path));
  if (!factory->IsInitialized())
    return false;

  state_subscriber_ = std::make_unique<
      igris_c_sdk::Subscriber<igris_c_sdk::msg::dds_::LowState_>>(
      config.state_topic, igris_c_sdk::QosProfile::SensorData());
  if (!state_subscriber_->init(
          [this](const auto &state) { lowStateCallback(state); }))
    return false;

  robot_state_subscriber_ = std::make_unique<
      igris_c_sdk::Subscriber<igris_c_sdk::msg::dds_::RobotState_>>(
      config.robot_state_topic, igris_c_sdk::QosProfile::Default());
  if (!robot_state_subscriber_->init(
          [this](const auto &state) { robotStateCallback(state); }))
    return false;

  command_publisher_ =
      std::make_unique<igris_c_sdk::Publisher<igris_c_sdk::msg::dds_::LowCmd_>>(
          config.command_topic, igris_c_sdk::QosProfile::SensorData());
  return command_publisher_->init();
}

bool RobotIo::waitForFirstState(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(state_mutex_);
  return state_cv_.wait_for(lock, timeout, [this] { return has_state_; });
}

bool RobotIo::snapshotState(
    igris_c_sdk::msg::dds_::LowState_ &state, uint64_t &sequence,
    std::chrono::steady_clock::time_point *received_at) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!has_state_)
    return false;
  state = latest_state_;
  sequence = state_sequence_;
  if (received_at)
    *received_at = state_received_at_;
  return true;
}

bool RobotIo::snapshotRobotMode(
    uint8_t &mode, uint64_t *low_mode_generation,
    std::chrono::steady_clock::time_point *low_mode_entered_at) const {
  std::lock_guard<std::mutex> lock(robot_state_mutex_);
  if (!has_robot_state_)
    return false;
  mode = latest_robot_state_.state();
  if (low_mode_generation)
    *low_mode_generation = low_mode_generation_;
  if (low_mode_entered_at)
    *low_mode_entered_at = low_mode_entered_at_;
  return true;
}

bool RobotIo::requestLowLevelMode(std::chrono::milliseconds timeout,
                                  std::string &error_message) {
  if (robotIsInLowLevelMode())
    return true;
  try {
    if (!control_client_) {
      control_client_ = std::make_unique<igris_c_sdk::IgrisC_Client>();
      control_client_->Init();
    }
    const auto response = control_client_->SendControlModeCommand(
        igris_c_sdk::srv::dds_::ControlModeCommandRequest_Request_Constants::
            CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL,
        "", false, static_cast<int>(timeout.count()));
    if (response.success() || robotIsInLowLevelMode())
      return true;
    error_message = response.message();
    if (error_message.empty())
      error_message = "control-mode request failed with error code " +
                      std::to_string(response.error_code());
    return false;
  } catch (const std::exception &error) {
    error_message = error.what();
    return false;
  }
}

bool RobotIo::enterLowLevelMode(std::chrono::milliseconds timeout,
                                std::string &error_message) {
  if (!requestLowLevelMode(timeout, error_message))
    return false;
  if (waitForLowLevelMode(timeout))
    return true;
  error_message = "low-level mode was not confirmed by RobotState";
  return false;
}

bool RobotIo::publish(const InferenceCommand &command) {
  if (!command_publisher_)
    return false;
  for (int i = 0; i < kModelDof; ++i) {
    if (!std::isfinite(static_cast<float>(command.q(i))) ||
        !std::isfinite(static_cast<float>(command.dq(i))) ||
        !std::isfinite(static_cast<float>(command.tau(i))) ||
        !std::isfinite(static_cast<float>(command.kp(i))) ||
        !std::isfinite(static_cast<float>(command.kd(i))))
      return false;
  }
  igris_c_sdk::msg::dds_::LowCmd_ message;
  message.header(BuildHeader());
  for (int i = 0; i < kKinematicGroupCount; ++i)
    message.kinematic_modes()[static_cast<std::size_t>(i)] =
        command.kinematic_modes[static_cast<std::size_t>(i)];
  for (int i = 0; i < kModelDof; ++i) {
    auto &motor = message.motors()[static_cast<std::size_t>(i)];
    motor.id(static_cast<uint16_t>(i));
    motor.q(static_cast<float>(command.q(i)));
    motor.dq(static_cast<float>(command.dq(i)));
    motor.tau(static_cast<float>(command.tau(i)));
    motor.kp(static_cast<float>(command.kp(i)));
    motor.kd(static_cast<float>(command.kd(i)));
  }
  return command_publisher_->write(message);
}

void RobotIo::lowStateCallback(const igris_c_sdk::msg::dds_::LowState_ &state) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_state_ = state;
    has_state_ = true;
    ++state_sequence_;
    state_received_at_ = std::chrono::steady_clock::now();
  }
  state_cv_.notify_all();
}

void RobotIo::robotStateCallback(
    const igris_c_sdk::msg::dds_::RobotState_ &state) {
  {
    std::lock_guard<std::mutex> lock(robot_state_mutex_);
    if (state.state() ==
            igris_c_sdk::msg::dds_::RobotState_Constants::ROBOT_STATE_LOW &&
        (!has_robot_state_ ||
         latest_robot_state_.state() !=
             igris_c_sdk::msg::dds_::RobotState_Constants::ROBOT_STATE_LOW)) {
      ++low_mode_generation_;
      low_mode_entered_at_ = std::chrono::steady_clock::now();
    }
    latest_robot_state_ = state;
    has_robot_state_ = true;
  }
  robot_state_cv_.notify_all();
}

bool RobotIo::robotIsInLowLevelMode() const {
  std::lock_guard<std::mutex> lock(robot_state_mutex_);
  return has_robot_state_ &&
         latest_robot_state_.state() ==
             igris_c_sdk::msg::dds_::RobotState_Constants::ROBOT_STATE_LOW;
}

bool RobotIo::waitForLowLevelMode(std::chrono::milliseconds timeout) const {
  std::unique_lock<std::mutex> lock(robot_state_mutex_);
  return robot_state_cv_.wait_for(lock, timeout, [this] {
    return has_robot_state_ &&
           latest_robot_state_.state() ==
               igris_c_sdk::msg::dds_::RobotState_Constants::ROBOT_STATE_LOW;
  });
}

std::string RobotIo::loadCycloneConfig(const std::string &path) {
  if (path.empty())
    return {};
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("failed to open CycloneDDS XML: " + path);
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

} // namespace igris_c_inference
