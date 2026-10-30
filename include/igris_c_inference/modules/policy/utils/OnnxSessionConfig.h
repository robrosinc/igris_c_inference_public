#pragma once

#include "onnxruntime_cxx_api.h"

namespace igris_c_inference::policy {

inline void configureLowLatencyCpuSession(Ort::SessionOptions &options) {
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  options.EnableMemPattern();
  options.EnableCpuMemArena();
  options.SetDeterministicCompute(true);
  options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  options.AddConfigEntry("session.inter_op.allow_spinning", "0");
}

} // namespace igris_c_inference::policy
