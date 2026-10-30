#pragma once

#include "igris_c_inference/core/pipeline_types.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace igris_c_inference::policy {

using NamedVectors = std::unordered_map<std::string, Eigen::VectorXd>;
using TensorMap = NamedVectors;
using ActionFrame = NamedVectors;

struct PolicySignal {
  Eigen::VectorXd value;
  std::vector<std::string> labels;
};

using SignalMap = std::unordered_map<std::string, PolicySignal>;

struct PolicyControlContext {
  SignalMap signals;
  uint64_t policyStep{0};
};

enum class ExecutionPhase { CONTROL, POLICY };

struct StepContext {
  const ProcessedState &status;
  ExecutionPhase phase{ExecutionPhase::POLICY};
  double dt{0.0};
  uint64_t controlStep{0};
  uint64_t policyStep{0};
  std::chrono::steady_clock::time_point now{std::chrono::steady_clock::now()};
};

enum class LifecyclePhase { STARTUP, RESET };

struct LifecycleContext {
  LifecyclePhase phase{LifecyclePhase::RESET};
  const ProcessedState *status{nullptr};
};

struct SignalDescriptor {
  std::string name;
  int size{0};
};

struct TermDescriptor {
  std::string name;
  std::string function;
  std::string binding;
  std::string updatePhase;
  int size{0};
  std::vector<std::string> dependencies;
  std::vector<SignalDescriptor> signals;
};

} // namespace igris_c_inference::policy
