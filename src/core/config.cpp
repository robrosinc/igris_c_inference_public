#include "igris_c_inference/core/config.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>

#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace igris_c_inference {
namespace {

[[noreturn]] void Fail(const std::string &where, const std::string &message) {
  throw std::runtime_error(where + ": " + message);
}

void CheckKeys(const YAML::Node &node,
               std::initializer_list<std::string_view> allowed,
               const std::string &where) {
  if (!node || !node.IsMap())
    Fail(where, "must be a map");
  std::unordered_set<std::string> valid;
  for (const auto key : allowed)
    valid.emplace(key);
  for (const auto &entry : node) {
    const std::string key = entry.first.as<std::string>();
    if (valid.find(key) == valid.end())
      Fail(where, "unknown key '" + key + "'");
  }
}

YAML::Node Require(const YAML::Node &node, const char *key,
                   const std::string &where) {
  if (!node[key])
    Fail(where, "missing required key '" + std::string(key) + "'");
  return node[key];
}

template <typename T>
T Value(const YAML::Node &node, const char *key, const std::string &where) {
  try {
    return Require(node, key, where).as<T>();
  } catch (const YAML::Exception &error) {
    Fail(where + "." + key, error.what());
  }
}

std::string ResolvePath(const std::filesystem::path &base,
                        const std::string &value, const std::string &where) {
  if (value.empty())
    Fail(where, "must not be empty");
  const std::filesystem::path path(value);
  return (path.is_absolute() ? path : base / path).lexically_normal().string();
}

Eigen::Vector3d ReadVector3(const YAML::Node &node, const std::string &where) {
  if (!node.IsSequence() || node.size() != 3)
    Fail(where, "must contain exactly 3 values");
  Eigen::Vector3d value;
  for (std::size_t i = 0; i < 3; ++i) {
    value(static_cast<Eigen::Index>(i)) = node[i].as<double>();
    if (!std::isfinite(value(static_cast<Eigen::Index>(i))))
      Fail(where, "values must be finite");
  }
  return value;
}

KinematicMode ParseMode(const YAML::Node &node, const std::string &where) {
  const std::string value = node.as<std::string>();
  if (value == "MS")
    return igris_c_sdk::msg::dds_::LowCmd_Constants::KINEMATIC_MODE_MS;
  if (value == "PJS")
    return igris_c_sdk::msg::dds_::LowCmd_Constants::KINEMATIC_MODE_PJS;
  Fail(where, "must be MS or PJS");
}

void RequireFinite(double value, const std::string &where) {
  if (!std::isfinite(value))
    Fail(where, "must be finite");
}

} // namespace

const RobotJointConfig &RobotConfig::joint(const std::string &name) const {
  for (const auto &candidate : joints)
    if (candidate.name == name)
      return candidate;
  throw std::runtime_error("unknown robot joint: " + name);
}

const RobotJointConfig &
RobotConfig::jointByModelName(const std::string &name) const {
  for (const auto &candidate : joints)
    if (candidate.model_joint == name)
      return candidate;
  throw std::runtime_error("unknown robot model joint: " + name);
}

const RobotFrameConfig &RobotConfig::frame(const std::string &name) const {
  const auto found = frames.find(name);
  if (found == frames.end())
    throw std::runtime_error("unknown robot frame: " + name);
  return found->second;
}

std::vector<int>
RobotConfig::resolveJointIndices(const std::vector<std::string> &names) const {
  std::unordered_set<std::string> seen;
  std::vector<int> indices;
  indices.reserve(names.size());
  for (const auto &name : names) {
    if (!seen.emplace(name).second)
      throw std::runtime_error("duplicate selected joint: " + name);
    indices.push_back(joint(name).robot_index);
  }
  return indices;
}

InferenceConfig LoadInferenceConfig(const std::string &path) {
  const YAML::Node root = YAML::LoadFile(path);
  CheckKeys(root, {"igris_c_inference"}, "params root");
  const YAML::Node config = Require(root, "igris_c_inference", "params root");
  CheckKeys(config, {"robot", "loop", "startup", "files"}, "igris_c_inference");
  const std::filesystem::path base = std::filesystem::path(path).parent_path();

  InferenceConfig output;
  const YAML::Node robot = Require(config, "robot", "igris_c_inference");
  CheckKeys(robot,
            {"domain_id", "namespace", "cyclonedds_xml_path", "state_topic",
             "robot_state_topic", "command_topic"},
            "igris_c_inference.robot");
  output.robot.domain_id =
      Value<int>(robot, "domain_id", "igris_c_inference.robot");
  output.robot.robot_namespace =
      Value<std::string>(robot, "namespace", "igris_c_inference.robot");
  const std::string xml = Value<std::string>(robot, "cyclonedds_xml_path",
                                             "igris_c_inference.robot");
  output.robot.cyclonedds_xml_path =
      xml.empty() ? std::string()
                  : ResolvePath(base, xml, "cyclonedds_xml_path");
  output.robot.state_topic =
      Value<std::string>(robot, "state_topic", "igris_c_inference.robot");
  output.robot.robot_state_topic =
      Value<std::string>(robot, "robot_state_topic", "igris_c_inference.robot");
  output.robot.command_topic =
      Value<std::string>(robot, "command_topic", "igris_c_inference.robot");
  if (output.robot.domain_id < 0 || output.robot.robot_namespace.empty() ||
      output.robot.state_topic.empty() ||
      output.robot.robot_state_topic.empty() ||
      output.robot.command_topic.empty())
    Fail("igris_c_inference.robot", "network values are invalid or empty");

  const YAML::Node loop = Require(config, "loop", "igris_c_inference");
  CheckKeys(loop, {"control_hz", "state_timeout_ms", "update_while_idle"},
            "igris_c_inference.loop");
  output.control_hz = Value<int>(loop, "control_hz", "igris_c_inference.loop");
  if (output.control_hz <= 0)
    Fail("igris_c_inference.loop", "control_hz must be positive");
  if (loop["update_while_idle"])
    output.update_while_idle =
        Value<bool>(loop, "update_while_idle", "igris_c_inference.loop");
  if (loop["state_timeout_ms"])
    output.state_timeout_ms =
        Value<int>(loop, "state_timeout_ms", "igris_c_inference.loop");
  if (output.state_timeout_ms <= 0)
    Fail("igris_c_inference.loop", "state_timeout_ms must be positive");

  const YAML::Node startup = Require(config, "startup", "igris_c_inference");
  CheckKeys(startup,
            {"first_state_timeout_ms", "enter_low_level_mode",
             "low_level_mode_timeout_ms", "command_timeout_ms"},
            "igris_c_inference.startup");
  if (startup["first_state_timeout_ms"])
    output.first_state_timeout_ms = Value<int>(
        startup, "first_state_timeout_ms", "igris_c_inference.startup");
  output.enter_low_level_mode =
      startup["enter_low_level_mode"]
          ? Value<bool>(startup, "enter_low_level_mode",
                        "igris_c_inference.startup")
          : true;
  output.low_level_mode_timeout_ms = Value<int>(
      startup, "low_level_mode_timeout_ms", "igris_c_inference.startup");
  if (startup["command_timeout_ms"])
    output.command_timeout_ms =
        Value<int>(startup, "command_timeout_ms", "igris_c_inference.startup");
  if (output.first_state_timeout_ms <= 0 ||
      output.low_level_mode_timeout_ms <= 0 || output.command_timeout_ms <= 0)
    Fail("igris_c_inference.startup", "timeouts must be positive");

  const YAML::Node files = Require(config, "files", "igris_c_inference");
  CheckKeys(files, {"robot", "policy_config", "policy"},
            "igris_c_inference.files");
  output.robot_path = ResolvePath(
      base, Value<std::string>(files, "robot", "igris_c_inference.files"),
      "files.robot");
  output.policy_config_path = ResolvePath(
      base,
      Value<std::string>(files, "policy_config", "igris_c_inference.files"),
      "files.policy_config");
  output.policy_path = ResolvePath(
      base, Value<std::string>(files, "policy", "igris_c_inference.files"),
      "files.policy");
  return output;
}

RobotConfig LoadRobotConfig(const std::string &path) {
  const YAML::Node root = YAML::LoadFile(path);
  CheckKeys(root, {"igris_c_inference_robot"}, "robot config root");
  const YAML::Node config =
      Require(root, "igris_c_inference_robot", "robot config root");
  CheckKeys(config, {"imu", "kinematic_modes", "robot_joints", "frames"},
            "igris_c_inference_robot");

  RobotConfig output;
  const YAML::Node imu = Require(config, "imu", "igris_c_inference_robot");
  CheckKeys(imu, {"base_to_imu_rpy", "gravity_direction"},
            "igris_c_inference_robot.imu");
  const Eigen::Vector3d rpy = ReadVector3(
      Require(imu, "base_to_imu_rpy", "igris_c_inference_robot.imu"),
      "igris_c_inference_robot.imu.base_to_imu_rpy");
  output.base_to_imu_rotation =
      (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  output.gravity_direction = ReadVector3(
      Require(imu, "gravity_direction", "igris_c_inference_robot.imu"),
      "igris_c_inference_robot.imu.gravity_direction");
  if (std::abs(output.gravity_direction.norm() - 1.0) > 1.0e-6)
    Fail("igris_c_inference_robot.imu.gravity_direction",
         "must be a unit vector");

  const YAML::Node modes =
      Require(config, "kinematic_modes", "igris_c_inference_robot");
  if (!modes.IsSequence() || modes.size() != kKinematicGroupCount)
    Fail("igris_c_inference_robot.kinematic_modes",
         "must contain exactly 5 entries");
  for (int i = 0; i < kKinematicGroupCount; ++i)
    output.kinematic_modes[static_cast<std::size_t>(i)] = ParseMode(
        modes[static_cast<std::size_t>(i)],
        "igris_c_inference_robot.kinematic_modes[" + std::to_string(i) + "]");

  const YAML::Node joints =
      Require(config, "robot_joints", "igris_c_inference_robot");
  if (!joints.IsSequence() || joints.size() != kModelDof)
    Fail("igris_c_inference_robot.robot_joints",
         "must contain exactly " + std::to_string(kModelDof) + " joints");
  std::array<bool, kModelDof> seen_indices{};
  std::unordered_set<std::string> seen_names;
  std::unordered_set<std::string> seen_model_joints;
  for (std::size_t i = 0; i < joints.size(); ++i) {
    const YAML::Node node = joints[i];
    const std::string where =
        "igris_c_inference_robot.robot_joints[" + std::to_string(i) + "]";
    CheckKeys(node,
              {"name", "model_joint", "robot_index", "position_min",
               "position_max", "default_position", "default_velocity",
               "default_torque", "default_kp", "default_kd"},
              where);
    RobotJointConfig joint;
    joint.name = Value<std::string>(node, "name", where);
    joint.model_joint = Value<std::string>(node, "model_joint", where);
    joint.robot_index = Value<int>(node, "robot_index", where);
    joint.position_min = Value<double>(node, "position_min", where);
    joint.position_max = Value<double>(node, "position_max", where);
    joint.default_position = Value<double>(node, "default_position", where);
    joint.default_velocity = Value<double>(node, "default_velocity", where);
    joint.default_torque = Value<double>(node, "default_torque", where);
    joint.default_kp = Value<double>(node, "default_kp", where);
    joint.default_kd = Value<double>(node, "default_kd", where);
    if (joint.name.empty() || joint.model_joint.empty())
      Fail(where, "joint names must not be empty");
    if (!seen_names.emplace(joint.name).second)
      Fail(where + ".name", "duplicate joint name");
    if (!seen_model_joints.emplace(joint.model_joint).second)
      Fail(where + ".model_joint", "duplicate model joint");
    if (joint.robot_index < 0 || joint.robot_index >= kModelDof ||
        seen_indices[static_cast<std::size_t>(joint.robot_index)])
      Fail(where + ".robot_index", "must uniquely cover the SDK joint range");
    seen_indices[static_cast<std::size_t>(joint.robot_index)] = true;
    const double values[] = {joint.position_min,     joint.position_max,
                             joint.default_position, joint.default_velocity,
                             joint.default_torque,   joint.default_kp,
                             joint.default_kd};
    for (const double value : values)
      RequireFinite(value, where);
    if (joint.position_min >= joint.position_max ||
        joint.default_position < joint.position_min ||
        joint.default_position > joint.position_max)
      Fail(where, "position limits or default position are invalid");
    if (joint.default_kp < 0.0 || joint.default_kd < 0.0)
      Fail(where, "default gains must not be negative");
    output.joints[static_cast<std::size_t>(joint.robot_index)] =
        std::move(joint);
  }

  const YAML::Node frames = config["frames"];
  if (frames) {
    if (!frames.IsMap())
      Fail("igris_c_inference_robot.frames", "must be a map");
    for (const auto &entry : frames) {
      RobotFrameConfig frame;
      frame.name = entry.first.as<std::string>();
      const std::string where = "igris_c_inference_robot.frames." + frame.name;
      const YAML::Node frame_node = entry.second;
      CheckKeys(frame_node, {"joints"}, where);
      const YAML::Node frame_joints = Require(frame_node, "joints", where);
      if (!frame_joints.IsSequence())
        Fail(where + ".joints", "must be a sequence");
      for (std::size_t i = 0; i < frame_joints.size(); ++i) {
        const YAML::Node node = frame_joints[i];
        const std::string joint_where =
            where + ".joints[" + std::to_string(i) + "]";
        CheckKeys(node, {"joint", "axis"}, joint_where);
        RobotFrameJointConfig joint;
        joint.joint = Value<std::string>(node, "joint", joint_where);
        joint.robot_index = output.joint(joint.joint).robot_index;
        joint.axis = ReadVector3(Require(node, "axis", joint_where),
                                 joint_where + ".axis");
        if (std::abs(joint.axis.norm() - 1.0) > 1.0e-6)
          Fail(joint_where + ".axis", "must be a unit vector");
        frame.joints.push_back(std::move(joint));
      }
      if (!output.frames.emplace(frame.name, std::move(frame)).second)
        Fail(where, "duplicate frame name");
    }
  }
  return output;
}

} // namespace igris_c_inference
