#include "igris_c_inference/modules/policy/runtime/PolicyRuntime.h"

#include "igris_c_inference/modules/policy/utils/OnnxSessionConfig.h"
#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace igris_c_inference::policy {

class PolicyRuntime::Impl {
public:
  explicit Impl(std::string runtime_label)
      : label(std::move(runtime_label)),
        environment(ORT_LOGGING_LEVEL_WARNING, label.c_str()) {}

  std::string label;
  Ort::Env environment;
  Ort::SessionOptions options;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::unique_ptr<Ort::Session> session;
  RlEnvConfig config;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<const char *> input_name_pointers;
  std::vector<const char *> output_name_pointers;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<float>> input_buffers;
  std::vector<std::vector<int64_t>> resolved_input_shapes;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;
  std::vector<ONNXTensorElementDataType> output_element_types;
  std::vector<std::size_t> output_element_counts;
  bool output_bindings_validated = false;
};

namespace {

template <typename T> std::string formatList(const std::vector<T> &values) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U)
      stream << ", ";
    stream << values[index];
  }
  stream << ']';
  return stream.str();
}

Eigen::Index firstNonFinite(const Eigen::VectorXd &values) {
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values(index)))
      return index;
  }
  return -1;
}

std::size_t elementCount(const std::vector<int64_t> &shape) {
  std::size_t count = 1U;
  for (const int64_t dimension : shape) {
    if (dimension <= 0)
      continue;
    if (count > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(dimension))
      throw std::runtime_error("invalid ONNX tensor shape");
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

std::vector<int64_t> resolveShape(const std::vector<int64_t> &original,
                                  std::size_t values) {
  std::vector<int64_t> shape = original;
  std::size_t dynamic_count = 0U;
  std::size_t dynamic_index = 0U;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] <= 0) {
      ++dynamic_count;
      dynamic_index = index;
    }
  }
  if (dynamic_count == 0U) {
    if (elementCount(shape) != values)
      throw std::runtime_error(
          "configured input size does not match ONNX tensor shape");
    return shape;
  }
  if (dynamic_count == 1U) {
    const std::size_t known = elementCount(shape);
    if (known == 0U || values % known != 0U)
      throw std::runtime_error("cannot resolve dynamic ONNX input shape");
    shape[dynamic_index] = static_cast<int64_t>(values / known);
    return shape;
  }
  if (shape.size() == 2U)
    return {1, static_cast<int64_t>(values)};
  return {static_cast<int64_t>(values)};
}

bool resolveWarmupElementCount(const std::vector<int64_t> &shape,
                               std::size_t &count) {
  if (shape.empty())
    return false;
  std::vector<int64_t> warmup_shape = shape;
  if (warmup_shape.front() <= 0 && warmup_shape.size() == 2U)
    warmup_shape.front() = 1;
  if (!std::all_of(warmup_shape.begin(), warmup_shape.end(),
                   [](int64_t dimension) { return dimension > 0; }))
    return false;
  count = elementCount(warmup_shape);
  return true;
}

template <typename RuntimeImpl>
void initializeInputTensors(RuntimeImpl &impl,
                            const std::vector<std::size_t> &element_counts) {
  impl.input_buffers.clear();
  impl.resolved_input_shapes.clear();
  impl.input_tensors.clear();
  impl.input_buffers.reserve(element_counts.size());
  impl.resolved_input_shapes.reserve(element_counts.size());
  impl.input_tensors.reserve(element_counts.size());
  for (std::size_t index = 0; index < element_counts.size(); ++index) {
    impl.input_buffers.emplace_back(element_counts[index], 0.0F);
    impl.resolved_input_shapes.push_back(
        resolveShape(impl.input_shapes[index], element_counts[index]));
    auto &buffer = impl.input_buffers.back();
    auto &shape = impl.resolved_input_shapes.back();
    impl.input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        impl.memory, buffer.data(), buffer.size(), shape.data(), shape.size()));
  }
}

template <typename RuntimeImpl> void runWithReusableOutputs(RuntimeImpl &impl) {
  if (impl.output_tensors.empty()) {
    impl.output_tensors = impl.session->Run(
        Ort::RunOptions{nullptr}, impl.input_name_pointers.data(),
        impl.input_tensors.data(), impl.input_tensors.size(),
        impl.output_name_pointers.data(), impl.output_name_pointers.size());
    if (impl.output_tensors.size() != impl.output_names.size())
      throw std::runtime_error(
          "ONNX runtime returned an unexpected output count");
    impl.output_element_types.clear();
    impl.output_element_counts.clear();
    impl.output_element_types.reserve(impl.output_tensors.size());
    impl.output_element_counts.reserve(impl.output_tensors.size());
    for (const auto &tensor : impl.output_tensors) {
      const auto info = tensor.GetTensorTypeAndShapeInfo();
      impl.output_element_types.push_back(info.GetElementType());
      impl.output_element_counts.push_back(info.GetElementCount());
    }
    return;
  }
  impl.session->Run(Ort::RunOptions{nullptr}, impl.input_name_pointers.data(),
                    impl.input_tensors.data(), impl.input_tensors.size(),
                    impl.output_name_pointers.data(),
                    impl.output_tensors.data(), impl.output_tensors.size());
}

template <typename RuntimeImpl>
void validateOutputBindings(const RuntimeImpl &impl) {
  for (const OutputBinding &binding : impl.config.outputs) {
    const auto iterator = std::find(
        impl.output_names.begin(), impl.output_names.end(), binding.tensorName);
    const std::size_t index = static_cast<std::size_t>(
        std::distance(impl.output_names.begin(), iterator));
    const std::size_t count = impl.output_element_counts[index];
    if (impl.output_element_types[index] != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("ONNX output tensor '" + binding.tensorName +
                               "' for binding '" + binding.name +
                               "' is not float");
    if (binding.offset > count || binding.size > count - binding.offset)
      throw std::runtime_error("output binding '" + binding.name +
                               "' exceeds tensor '" + binding.tensorName + "'");
  }
}

} // namespace

PolicyRuntime::PolicyRuntime(std::string label)
    : impl_(std::make_unique<Impl>(std::move(label))) {}
PolicyRuntime::~PolicyRuntime() = default;

int PolicyRuntime::configure(const std::string &model_path,
                             const RlEnvConfig &config, std::string &error) {
  try {
    if (model_path.empty() || !std::filesystem::exists(model_path))
      throw std::runtime_error("policy model does not exist: " + model_path);
    impl_->options = Ort::SessionOptions{};
    configureLowLatencyCpuSession(impl_->options);
    impl_->session = std::make_unique<Ort::Session>(
        impl_->environment, model_path.c_str(), impl_->options);
    impl_->config = config;
    impl_->input_names.clear();
    impl_->output_names.clear();
    impl_->input_shapes.clear();
    for (std::size_t index = 0; index < impl_->session->GetInputCount();
         ++index) {
      auto name =
          impl_->session->GetInputNameAllocated(index, impl_->allocator);
      impl_->input_names.emplace_back(name.get());
    }
    if (impl_->input_names != config.inputNames)
      throw std::runtime_error(
          "configured input names " + formatList(config.inputNames) +
          " do not match ONNX input names " + formatList(impl_->input_names));
    for (std::size_t index = 0; index < impl_->session->GetInputCount();
         ++index) {
      const Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(index);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error(
            "ONNX input must be a float tensor: " + impl_->input_names[index] +
            " (type=" +
            std::to_string(static_cast<int>(info.GetElementType())) + ")");
      const std::vector<int64_t> modelShape = info.GetShape();
      if (config.inputShapes.empty()) {
        impl_->input_shapes.push_back(modelShape);
        continue;
      }
      const auto &configuredShape = config.inputShapes[index];
      if (configuredShape.size() != modelShape.size())
        throw std::runtime_error(
            "configured input shape rank does not match ONNX input '" +
            impl_->input_names[index] + "'");
      for (std::size_t dimension = 0; dimension < modelShape.size();
           ++dimension) {
        if (modelShape[dimension] > 0 &&
            modelShape[dimension] != configuredShape[dimension])
          throw std::runtime_error(
              "configured input shape does not match ONNX input '" +
              impl_->input_names[index] + "'");
      }
      impl_->input_shapes.push_back(configuredShape);
    }
    for (std::size_t index = 0; index < impl_->session->GetOutputCount();
         ++index) {
      auto name =
          impl_->session->GetOutputNameAllocated(index, impl_->allocator);
      impl_->output_names.emplace_back(name.get());
    }
    for (const OutputBinding &binding : impl_->config.outputs) {
      if (std::find(impl_->output_names.begin(), impl_->output_names.end(),
                    binding.tensorName) == impl_->output_names.end())
        throw std::runtime_error("configured output tensor is missing: " +
                                 binding.tensorName);
    }
    impl_->input_name_pointers.clear();
    impl_->output_name_pointers.clear();
    for (const auto &name : impl_->input_names)
      impl_->input_name_pointers.push_back(name.c_str());
    for (const auto &name : impl_->output_names)
      impl_->output_name_pointers.push_back(name.c_str());
    impl_->input_buffers.clear();
    impl_->resolved_input_shapes.clear();
    impl_->input_tensors.clear();
    impl_->output_tensors.clear();
    impl_->output_element_types.clear();
    impl_->output_element_counts.clear();
    impl_->output_bindings_validated = false;

    std::vector<std::size_t> element_counts;
    element_counts.reserve(impl_->input_shapes.size());
    bool can_warm_up = !impl_->input_shapes.empty();
    for (const auto &shape : impl_->input_shapes) {
      std::size_t count = 0U;
      if (!resolveWarmupElementCount(shape, count)) {
        can_warm_up = false;
        break;
      }
      element_counts.push_back(count);
    }
    if (can_warm_up) {
      initializeInputTensors(*impl_, element_counts);
      runWithReusableOutputs(*impl_);
      validateOutputBindings(*impl_);
      impl_->output_bindings_validated = true;
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    impl_->session.reset();
    error =
        "configure stage for '" + model_path + "' failed: " + exception.what();
    return -1;
  }
}

void PolicyRuntime::reset() {}

std::vector<std::pair<std::string, std::size_t>>
PolicyRuntime::expectedInputSizes() const {
  std::vector<std::pair<std::string, std::size_t>> result;
  result.reserve(impl_->input_names.size());
  for (std::size_t index = 0; index < impl_->input_names.size(); ++index) {
    std::size_t count = 0U;
    if (resolveWarmupElementCount(impl_->input_shapes[index], count))
      result.emplace_back(impl_->input_names[index], count);
  }
  return result;
}

int PolicyRuntime::run(const TensorMap &inputs, TensorMap &outputs,
                       std::string &error) {
  if (!impl_->session) {
    error = "run requested before model configuration";
    return -1;
  }
  try {
    if (impl_->input_tensors.empty()) {
      std::vector<std::size_t> element_counts;
      element_counts.reserve(impl_->input_names.size());
      for (const auto &name : impl_->input_names) {
        const auto input = inputs.find(name);
        if (input == inputs.end())
          throw std::runtime_error("missing policy input '" + name + "'");
        element_counts.push_back(
            static_cast<std::size_t>(input->second.size()));
      }
      initializeInputTensors(*impl_, element_counts);
    }
    for (std::size_t index = 0; index < impl_->input_names.size(); ++index) {
      const auto input = inputs.find(impl_->input_names[index]);
      if (input == inputs.end())
        throw std::runtime_error("missing policy input '" +
                                 impl_->input_names[index] + "'");
      const Eigen::Index invalid_index = firstNonFinite(input->second);
      if (invalid_index >= 0)
        throw std::runtime_error("policy input '" + impl_->input_names[index] +
                                 "' is non-finite at index " +
                                 std::to_string(invalid_index));
      auto &buffer = impl_->input_buffers[index];
      if (static_cast<std::size_t>(input->second.size()) != buffer.size())
        throw std::runtime_error("policy input '" + impl_->input_names[index] +
                                 "' size changed");
      for (Eigen::Index value = 0; value < input->second.size(); ++value)
        buffer[static_cast<std::size_t>(value)] =
            static_cast<float>(input->second(value));
    }
    runWithReusableOutputs(*impl_);
    if (!impl_->output_bindings_validated) {
      validateOutputBindings(*impl_);
      impl_->output_bindings_validated = true;
    }
    outputs.clear();
    for (const OutputBinding &binding : impl_->config.outputs) {
      const auto iterator =
          std::find(impl_->output_names.begin(), impl_->output_names.end(),
                    binding.tensorName);
      const std::size_t index = static_cast<std::size_t>(
          std::distance(impl_->output_names.begin(), iterator));
      const float *data = impl_->output_tensors[index].GetTensorData<float>();
      Eigen::VectorXd &result = outputs[binding.name];
      result.resize(static_cast<Eigen::Index>(binding.size));
      for (std::size_t value = 0; value < binding.size; ++value)
        result(static_cast<Eigen::Index>(value)) = data[binding.offset + value];
      const Eigen::Index invalid_index = firstNonFinite(result);
      if (invalid_index >= 0)
        throw std::runtime_error("policy output binding '" + binding.name +
                                 "' is non-finite at index " +
                                 std::to_string(invalid_index));
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = "inference failed: " + std::string(exception.what());
    return -1;
  }
}

} // namespace igris_c_inference::policy
