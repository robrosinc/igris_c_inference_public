#pragma once

#include "igris_c_inference/modules/policy/utils/PolicyPipelineTypes.h"

#include <chrono>
#include <cstdint>
#include <mutex>

namespace igris_c_inference {

struct AsyncPolicyInferenceResult {
  uint64_t sequence = 0;
  policy::ActionFrame actions;
};

class AsyncPolicyModule {
public:
  virtual ~AsyncPolicyModule() = default;

  virtual std::chrono::microseconds inferencePeriod() const = 0;
  virtual int runInferenceWorkerStep(const ProcessedState &status) = 0;

protected:
  void publishInferenceResult(const policy::ActionFrame &actions) {
    std::lock_guard<std::mutex> lock(inferenceResultMutex_);
    latestInferenceResult_.actions = actions;
    ++latestInferenceResult_.sequence;
  }

  void resetInferenceResult() {
    std::lock_guard<std::mutex> lock(inferenceResultMutex_);
    latestInferenceResult_ = AsyncPolicyInferenceResult{};
  }

  bool consumeLatestInferenceResult(uint64_t &lastSequence,
                                    policy::ActionFrame &actions) const {
    std::lock_guard<std::mutex> lock(inferenceResultMutex_);
    if (latestInferenceResult_.sequence == 0 ||
        latestInferenceResult_.sequence == lastSequence)
      return false;
    actions = latestInferenceResult_.actions;
    lastSequence = latestInferenceResult_.sequence;
    return true;
  }

private:
  mutable std::mutex inferenceResultMutex_;
  AsyncPolicyInferenceResult latestInferenceResult_;
};

} // namespace igris_c_inference
