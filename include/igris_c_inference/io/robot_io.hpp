#pragma once

#include "igris_c_inference/core/config.hpp"
#include "igris_c_inference/core/types.hpp"

#include "igris_c_sdk/igris_c_client.hpp"
#include "igris_c_sdk/igris_c_msgs.hpp"
#include "igris_c_sdk/publisher.hpp"
#include "igris_c_sdk/subscriber.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace igris_c_inference {

class RobotIo {
public:
  bool initialize(const RobotNetworkConfig &config);
  bool waitForFirstState(std::chrono::milliseconds timeout);
  bool snapshotState(
      igris_c_sdk::msg::dds_::LowState_ &state, uint64_t &sequence,
      std::chrono::steady_clock::time_point *received_at = nullptr) const;
  bool snapshotRobotMode(uint8_t &mode, uint64_t *low_mode_generation = nullptr,
                         std::chrono::steady_clock::time_point
                             *low_mode_entered_at = nullptr) const;
  bool requestLowLevelMode(std::chrono::milliseconds timeout,
                           std::string &error_message);
  bool enterLowLevelMode(std::chrono::milliseconds timeout,
                         std::string &error_message);
  bool publish(const InferenceCommand &command);

private:
  void lowStateCallback(const igris_c_sdk::msg::dds_::LowState_ &state);
  void robotStateCallback(const igris_c_sdk::msg::dds_::RobotState_ &state);
  bool robotIsInLowLevelMode() const;
  bool waitForLowLevelMode(std::chrono::milliseconds timeout) const;
  static std::string loadCycloneConfig(const std::string &path);

  mutable std::mutex state_mutex_;
  std::condition_variable state_cv_;
  igris_c_sdk::msg::dds_::LowState_ latest_state_;
  bool has_state_ = false;
  uint64_t state_sequence_ = 0;
  std::chrono::steady_clock::time_point state_received_at_{};

  mutable std::mutex robot_state_mutex_;
  mutable std::condition_variable robot_state_cv_;
  igris_c_sdk::msg::dds_::RobotState_ latest_robot_state_;
  bool has_robot_state_ = false;
  uint64_t low_mode_generation_ = 0;
  std::chrono::steady_clock::time_point low_mode_entered_at_{};

  std::unique_ptr<igris_c_sdk::Subscriber<igris_c_sdk::msg::dds_::LowState_>>
      state_subscriber_;
  std::unique_ptr<igris_c_sdk::Subscriber<igris_c_sdk::msg::dds_::RobotState_>>
      robot_state_subscriber_;
  std::unique_ptr<igris_c_sdk::Publisher<igris_c_sdk::msg::dds_::LowCmd_>>
      command_publisher_;
  std::unique_ptr<igris_c_sdk::IgrisC_Client> control_client_;
};

} // namespace igris_c_inference
