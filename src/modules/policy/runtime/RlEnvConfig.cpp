#include "igris_c_inference/modules/policy/runtime/RlEnvConfig.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <set>
#include <utility>

namespace igris_c_inference::policy {
namespace {

bool hasOnlyKeys(const YAML::Node &node, const std::set<std::string> &allowed,
                 std::string &error, const std::string &where) {
  if (!node || !node.IsMap()) {
    error = where + " must be a map";
    return false;
  }
  for (const auto &entry : node) {
    const std::string key = entry.first.as<std::string>();
    if (allowed.count(key) == 0U) {
      error = where + " contains unknown key '" + key + "'";
      return false;
    }
  }
  return true;
}

std::string resolvePath(const std::filesystem::path &base,
                        const std::string &value) {
  const std::filesystem::path path(value);
  return (path.is_absolute() ? path : base / path).lexically_normal().string();
}

} // namespace

int loadRlEnvConfig(const std::string &path, RlEnvConfig &config,
                    std::string &error) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    if (!hasOnlyKeys(root, {"igris_c_inference_policy"}, error, "policy root"))
      return -1;
    const YAML::Node node = root["igris_c_inference_policy"];
    if (!hasOnlyKeys(node,
                     {"rate_hz", "files", "inputs", "input_shapes", "outputs"},
                     error, "igris_c_inference_policy"))
      return -1;
    if (!node["rate_hz"] || !node["files"] || !node["inputs"] ||
        !node["outputs"]) {
      error = "igris_c_inference_policy requires rate_hz, files, inputs, and "
              "outputs";
      return -1;
    }

    RlEnvConfig parsed;
    parsed.rateHz = node["rate_hz"].as<int>();
    if (parsed.rateHz <= 0) {
      error = "igris_c_inference_policy.rate_hz must be positive";
      return -1;
    }

    const YAML::Node files = node["files"];
    if (!hasOnlyKeys(
            files, {"command", "observation", "action", "termination", "event"},
            error, "igris_c_inference_policy.files"))
      return -1;
    if (!files["command"] || !files["observation"] || !files["action"] ||
        !files["termination"] || !files["event"]) {
      error = "igris_c_inference_policy.files requires command, observation, "
              "action, termination, and event";
      return -1;
    }
    const auto base = std::filesystem::path(path).parent_path();
    parsed.commandPath = resolvePath(base, files["command"].as<std::string>());
    parsed.observationPath =
        resolvePath(base, files["observation"].as<std::string>());
    parsed.actionPath = resolvePath(base, files["action"].as<std::string>());
    parsed.terminationPath =
        resolvePath(base, files["termination"].as<std::string>());
    parsed.eventPath = resolvePath(base, files["event"].as<std::string>());

    const YAML::Node inputs = node["inputs"];
    if (!inputs.IsSequence() || inputs.size() == 0U) {
      error = "igris_c_inference_policy.inputs must be a non-empty sequence";
      return -1;
    }
    std::set<std::string> names;
    for (const auto &input : inputs) {
      if (!input.IsScalar()) {
        error = "igris_c_inference_policy.inputs entries must be names";
        return -1;
      }
      const std::string name = input.as<std::string>();
      if (name.empty() || !names.insert(name).second) {
        error = "igris_c_inference_policy.inputs must contain unique non-empty "
                "names";
        return -1;
      }
      parsed.inputNames.push_back(name);
    }
    if (node["input_shapes"]) {
      const YAML::Node inputShapes = node["input_shapes"];
      if (!inputShapes.IsSequence() ||
          inputShapes.size() != parsed.inputNames.size()) {
        error = "igris_c_inference_policy.input_shapes must match inputs";
        return -1;
      }
      for (const auto &shapeNode : inputShapes) {
        if (!shapeNode.IsSequence() || shapeNode.size() == 0U) {
          error = "igris_c_inference_policy.input_shapes entries must be "
                  "non-empty sequences";
          return -1;
        }
        std::vector<int64_t> shape;
        for (const auto &dimension : shapeNode) {
          const int64_t value = dimension.as<int64_t>();
          if (value <= 0) {
            error = "igris_c_inference_policy.input_shapes dimensions must be "
                    "positive";
            return -1;
          }
          shape.push_back(value);
        }
        parsed.inputShapes.push_back(std::move(shape));
      }
    }

    const YAML::Node outputs = node["outputs"];
    if (!outputs.IsSequence() || outputs.size() == 0U) {
      error = "igris_c_inference_policy.outputs must be a non-empty sequence";
      return -1;
    }
    names.clear();
    for (const auto &output : outputs) {
      if (!hasOnlyKeys(output, {"name", "tensor", "offset", "size"}, error,
                       "igris_c_inference_policy.outputs[]"))
        return -1;
      if (!output["name"] || !output["tensor"] || !output["offset"] ||
          !output["size"]) {
        error = "each output requires name, tensor, offset, and size";
        return -1;
      }
      OutputBinding binding;
      binding.name = output["name"].as<std::string>();
      binding.tensorName = output["tensor"].as<std::string>();
      binding.offset = output["offset"].as<std::size_t>();
      binding.size = output["size"].as<std::size_t>();
      if (binding.name.empty() || binding.tensorName.empty() ||
          binding.size == 0U || !names.insert(binding.name).second) {
        error = "outputs require unique exposed names, non-empty tensor names, "
                "and non-zero sizes";
        return -1;
      }
      parsed.outputs.push_back(std::move(binding));
    }
    config = std::move(parsed);
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

} // namespace igris_c_inference::policy
