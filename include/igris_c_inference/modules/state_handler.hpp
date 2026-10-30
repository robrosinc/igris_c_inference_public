#pragma once

#include "igris_c_inference/core/config.hpp"
#include "igris_c_inference/core/pipeline_types.hpp"

#include "igris_c_sdk/igris_c_msgs.hpp"

#include <cstdint>

namespace igris_c_inference {

class StateHandler {
public:
  void configure(const RobotConfig &robot);
  bool update(const igris_c_sdk::msg::dds_::LowState_ &raw, uint64_t sequence,
              ProcessedState &state) const;

private:
  RobotConfig robot_;
};

} // namespace igris_c_inference
