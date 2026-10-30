#include "igris_c_inference/modules/policy/mdp/commands/utils/MotionReferenceSource.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"

#include "onnxruntime_cxx_api.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace igris_c_inference::policy::command {
namespace {

template <typename T>
T Required(const YAML::Node &node, const char *key, const std::string &where) {
  if (!node || !node[key])
    throw std::runtime_error(where + ": missing '" + key + "'");
  return node[key].as<T>();
}

std::string ResolvePath(const FactoryContext &context,
                        const std::string &value) {
  const std::filesystem::path path(value);
  return (path.is_absolute()
              ? path
              : std::filesystem::path(context.profileDirectory) / path)
      .lexically_normal()
      .string();
}

std::vector<std::string> SplitCommaSeparated(const std::string &value) {
  std::vector<std::string> values;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const auto begin =
        std::find_if_not(token.begin(), token.end(), [](unsigned char ch) {
          return std::isspace(ch) != 0;
        });
    const auto end =
        std::find_if_not(token.rbegin(), token.rend(), [](unsigned char ch) {
          return std::isspace(ch) != 0;
        }).base();
    if (begin < end)
      values.emplace_back(begin, end);
  }
  return values;
}

std::string Metadata(const Ort::ModelMetadata &metadata,
                     Ort::AllocatorWithDefaultOptions &allocator,
                     const char *key) {
  Ort::AllocatedStringPtr value =
      metadata.LookupCustomMetadataMapAllocated(key, allocator);
  return value ? std::string(value.get()) : std::string();
}

std::size_t PositiveSize(const std::string &value, const std::string &label) {
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed == 0)
    throw std::runtime_error("invalid " + label + " metadata: " + value);
  return static_cast<std::size_t>(parsed);
}

double PositiveDouble(const std::string &value, const std::string &label) {
  char *end = nullptr;
  const double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0' || !std::isfinite(parsed) ||
      parsed <= 0.0)
    throw std::runtime_error("invalid " + label + " metadata: " + value);
  return parsed;
}

std::size_t ElementCount(const Ort::Value &value) {
  return value.GetTensorTypeAndShapeInfo().GetElementCount();
}

class OnnxMotionSource final : public MotionReferenceSource {
public:
  OnnxMotionSource(const YAML::Node &params, const FactoryContext &context,
                   const std::string &where)
      : env_(ORT_LOGGING_LEVEL_WARNING, "igris_c_motion_reference") {
    path_ = ResolvePath(
        context, Required<std::string>(params, "path", where + ".params"));
    loop_ = params["loop"] ? params["loop"].as<bool>() : true;
    const std::string quaternion_order =
        params["quaternion_order"]
            ? params["quaternion_order"].as<std::string>()
            : "wxyz";
    if (quaternion_order == "wxyz")
      quaternion_order_ = QuaternionOrder::Wxyz;
    else if (quaternion_order == "xyzw")
      quaternion_order_ = QuaternionOrder::Xyzw;
    else
      throw std::runtime_error(where +
                               ".source.params.quaternion_order must be "
                               "wxyz or xyzw");
    if (!std::filesystem::exists(path_))
      throw std::runtime_error("motion ONNX does not exist: " + path_);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options_.AddConfigEntry("session.use_deterministic_compute", "1");
    session_ = std::make_unique<Ort::Session>(env_, path_.c_str(), options_);
    loadContract();
    if (std::abs(frame_rate_ - context.policyHz) > 1.0e-6)
      throw std::runtime_error(
          "motion ONNX motion_fps must match the configured policy_hz");
  }

  bool start() override { return true; }
  void stop() override {}
  void reset() override {
    cached_.clear();
    cache_start_ = 0;
    cache_valid_ = false;
  }
  bool waitUntilReady(std::size_t, std::chrono::milliseconds) override {
    return true;
  }
  bool stale(Clock::time_point, std::chrono::milliseconds) const override {
    return false;
  }

  bool frames(uint64_t policy_step, std::size_t frame_count,
              std::vector<MotionReferenceFrame> &output) override {
    if (frame_count == 0)
      return false;
    if (cache_valid_ && cached_.size() == frame_count &&
        policy_step == cache_start_) {
      // A failed policy step is prepared again from the identical window.
    } else if (cache_valid_ && cached_.size() == frame_count &&
               policy_step == cache_start_ + 1) {
      cached_.pop_front();
      cached_.push_back(run(policy_step + frame_count - 1));
    } else {
      cached_.clear();
      for (std::size_t i = 0; i < frame_count; ++i)
        cached_.push_back(run(policy_step + i));
    }
    cache_start_ = policy_step;
    cache_valid_ = true;
    output.assign(cached_.begin(), cached_.end());
    return true;
  }

private:
  void loadContract() {
    if (session_->GetInputCount() != 1)
      throw std::runtime_error("motion ONNX must have exactly one input");
    Ort::AllocatedStringPtr input =
        session_->GetInputNameAllocated(0, allocator_);
    input_name_ = input.get();
    if (input_name_ != "time_step")
      throw std::runtime_error("motion ONNX input must be named time_step");
    const Ort::TypeInfo input_type = session_->GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
      throw std::runtime_error("motion ONNX time_step input must be int64");
    for (std::size_t i = 0; i < session_->GetOutputCount(); ++i) {
      Ort::AllocatedStringPtr name =
          session_->GetOutputNameAllocated(i, allocator_);
      output_names_.emplace_back(name.get());
      output_indices_.emplace(output_names_.back(), i);
      const Ort::TypeInfo output_type = session_->GetOutputTypeInfo(i);
      const auto output_info = output_type.GetTensorTypeAndShapeInfo();
      if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("motion ONNX outputs must be float tensors");
    }
    for (const char *required : {"joint_pos", "joint_vel", "body_quat_w"})
      if (output_indices_.find(required) == output_indices_.end())
        throw std::runtime_error("motion ONNX is missing output '" +
                                 std::string(required) + "'");
    Ort::ModelMetadata metadata = session_->GetModelMetadata();
    joint_names_ =
        SplitCommaSeparated(Metadata(metadata, allocator_, "joint_names"));
    body_names_ =
        SplitCommaSeparated(Metadata(metadata, allocator_, "body_names"));
    anchor_body_name_ = Metadata(metadata, allocator_, "local_frame_body_name");
    const std::string frames =
        Metadata(metadata, allocator_, "motion_num_frames");
    const std::string fps = Metadata(metadata, allocator_, "motion_fps");
    if (joint_names_.empty() || body_names_.empty() || frames.empty() ||
        fps.empty())
      throw std::runtime_error(
          "motion ONNX metadata requires joint_names, body_names, and "
          "motion_num_frames and motion_fps");
    frame_count_ = PositiveSize(frames, "motion_num_frames");
    frame_rate_ = PositiveDouble(fps, "motion_fps");
    if (anchor_body_name_.empty() &&
        std::find(body_names_.begin(), body_names_.end(), "base_link") !=
            body_names_.end())
      anchor_body_name_ = "base_link";
    const auto anchor =
        std::find(body_names_.begin(), body_names_.end(), anchor_body_name_);
    if (anchor != body_names_.end())
      anchor_body_index_ =
          static_cast<std::size_t>(std::distance(body_names_.begin(), anchor));
  }

  MotionReferenceFrame run(uint64_t step) {
    const int64_t index = static_cast<int64_t>(
        loop_ ? step % frame_count_
              : std::min<uint64_t>(step, frame_count_ - 1));
    std::array<int64_t, 1> data{index};
    std::array<int64_t, 2> shape{1, 1};
    Ort::Value input = Ort::Value::CreateTensor<int64_t>(
        memory_, data.data(), data.size(), shape.data(), shape.size());
    std::vector<const char *> output_ptrs;
    for (const auto &name : output_names_)
      output_ptrs.push_back(name.c_str());
    const char *input_ptr = input_name_.c_str();
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, &input_ptr, &input,
                                 1, output_ptrs.data(), output_ptrs.size());
    MotionReferenceFrame frame;
    frame.sequence = step;
    frame.joint_names = joint_names_;
    copyVector(outputs.at(output_indices_.at("joint_pos")), joint_names_.size(),
               frame.joint_position, "joint_pos");
    copyVector(outputs.at(output_indices_.at("joint_vel")), joint_names_.size(),
               frame.joint_velocity, "joint_vel");
    const Ort::Value &quat = outputs.at(output_indices_.at("body_quat_w"));
    if (ElementCount(quat) < body_names_.size() * 4)
      throw std::runtime_error("body_quat_w is smaller than body_names");
    frame.body_names = body_names_;
    const float *q = quat.GetTensorData<float>();
    frame.world_from_body.reserve(body_names_.size());
    for (std::size_t i = 0; i < body_names_.size(); ++i) {
      const std::size_t offset = i * 4;
      Eigen::Quaterniond value =
          quaternion_order_ == QuaternionOrder::Wxyz
              ? Eigen::Quaterniond(q[offset], q[offset + 1], q[offset + 2],
                                   q[offset + 3])
              : Eigen::Quaterniond(q[offset + 3], q[offset], q[offset + 1],
                                   q[offset + 2]);
      if (!std::isfinite(value.norm()) || value.norm() < 1.0e-12)
        throw std::runtime_error("motion ONNX contains an invalid quaternion");
      frame.world_from_body.push_back(value.normalized());
    }
    frame.anchor_body_name = anchor_body_name_;
    if (hasOutput("body_pos_b")) {
      const Ort::Value &position = outputs.at(output_indices_.at("body_pos_b"));
      if (ElementCount(position) < body_names_.size() * 3)
        throw std::runtime_error("body_pos_b is smaller than body_names");
      const float *data = position.GetTensorData<float>();
      frame.anchor_from_body_position.reserve(body_names_.size());
      for (std::size_t i = 0; i < body_names_.size(); ++i)
        frame.anchor_from_body_position.emplace_back(
            static_cast<double>(data[i * 3]),
            static_cast<double>(data[i * 3 + 1]),
            static_cast<double>(data[i * 3 + 2]));
    }
    if (anchor_body_index_ < body_names_.size()) {
      copyAnchorVector(outputs, "body_pos_w", frame.world_anchor_position,
                       frame.world_anchor_position_valid);
      copyAnchorVector(outputs, "body_lin_vel_b", frame.anchor_linear_velocity,
                       frame.anchor_linear_velocity_valid);
      copyAnchorVector(outputs, "body_ang_vel_b", frame.anchor_angular_velocity,
                       frame.anchor_angular_velocity_valid);
    }
    return frame;
  }

  bool hasOutput(const std::string &name) const {
    return output_indices_.find(name) != output_indices_.end();
  }

  void copyAnchorVector(const std::vector<Ort::Value> &outputs,
                        const std::string &name, Eigen::Vector3d &target,
                        bool &valid) const {
    if (!hasOutput(name))
      return;
    const Ort::Value &value = outputs.at(output_indices_.at(name));
    const std::size_t offset = anchor_body_index_ * 3;
    if (ElementCount(value) < offset + 3)
      throw std::runtime_error("motion ONNX output '" + name +
                               "' is smaller than its anchor body");
    const float *data = value.GetTensorData<float>() + offset;
    target = Eigen::Vector3d(static_cast<double>(data[0]),
                             static_cast<double>(data[1]),
                             static_cast<double>(data[2]));
    valid = target.allFinite();
  }

  static void copyVector(const Ort::Value &value, std::size_t size,
                         Eigen::VectorXd &target, const std::string &name) {
    if (ElementCount(value) < size)
      throw std::runtime_error("motion ONNX output '" + name +
                               "' is too small");
    target.resize(static_cast<Eigen::Index>(size));
    const float *data = value.GetTensorData<float>();
    for (std::size_t i = 0; i < size; ++i)
      target(static_cast<Eigen::Index>(i)) = data[i];
  }

  std::string path_;
  bool loop_ = true;
  std::size_t frame_count_ = 0;
  double frame_rate_ = 0.0;
  std::string input_name_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, std::size_t> output_indices_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> body_names_;
  std::string anchor_body_name_;
  enum class QuaternionOrder { Wxyz, Xyzw };
  QuaternionOrder quaternion_order_ = QuaternionOrder::Wxyz;
  std::size_t anchor_body_index_ = std::numeric_limits<std::size_t>::max();
  std::deque<MotionReferenceFrame> cached_;
  uint64_t cache_start_ = 0;
  bool cache_valid_ = false;
  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo memory_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::unique_ptr<Ort::Session> session_;
};

bool Numeric(char value) {
  return std::isdigit(static_cast<unsigned char>(value)) || value == '-' ||
         value == '+' || value == '.' || value == 'e' || value == 'E';
}

bool Identifier(char value) {
  return std::isalnum(static_cast<unsigned char>(value)) || value == '_';
}

std::size_t FindKey(const std::string &payload, const std::string &key) {
  std::size_t position = payload.find(key);
  while (position != std::string::npos) {
    const bool before = position == 0 || !Identifier(payload[position - 1]);
    const std::size_t after_position = position + key.size();
    const bool after = after_position == payload.size() ||
                       !Identifier(payload[after_position]);
    if (before && after)
      return position;
    position = payload.find(key, position + 1);
  }
  return std::string::npos;
}

bool ArrayBounds(const std::string &payload, const std::string &key,
                 std::size_t &begin, std::size_t &end) {
  const std::size_t key_position = FindKey(payload, key);
  if (key_position == std::string::npos)
    return false;
  begin = payload.find('[', key_position);
  if (begin == std::string::npos)
    return false;
  int depth = 0;
  for (std::size_t i = begin; i < payload.size(); ++i) {
    if (payload[i] == '[')
      ++depth;
    else if (payload[i] == ']' && --depth == 0) {
      end = i;
      return true;
    }
  }
  return false;
}

bool NumberArray(const std::string &payload, const std::string &key,
                 std::vector<double> &output) {
  std::size_t begin = 0, end = 0;
  if (!ArrayBounds(payload, key, begin, end))
    return false;
  output.clear();
  std::string token;
  for (std::size_t i = begin + 1; i < end; ++i) {
    if (Numeric(payload[i]))
      token.push_back(payload[i]);
    else if (!token.empty()) {
      output.push_back(std::stod(token));
      token.clear();
    }
  }
  if (!token.empty())
    output.push_back(std::stod(token));
  return !output.empty();
}

bool StringArray(const std::string &payload, const std::string &key,
                 std::vector<std::string> &output) {
  std::size_t begin = 0, end = 0;
  if (!ArrayBounds(payload, key, begin, end))
    return false;
  output.clear();
  bool quoted = false;
  char quote = 0;
  std::string token;
  for (std::size_t i = begin + 1; i < end; ++i) {
    if (!quoted && (payload[i] == '\'' || payload[i] == '"')) {
      quoted = true;
      quote = payload[i];
    } else if (quoted && payload[i] == quote) {
      output.push_back(token);
      token.clear();
      quoted = false;
    } else if (quoted) {
      token.push_back(payload[i]);
    }
  }
  return !output.empty();
}

class RosMotionSource final : public MotionReferenceSource {
public:
  RosMotionSource(const YAML::Node &params, const FactoryContext &context,
                  const std::string &where) {
    topic_ = Required<std::string>(params, "topic", where + ".params");
    node_name_ = params["node_name"] ? params["node_name"].as<std::string>()
                                     : "igris_c_sonic_motion_reference";
    domain_id_ = params["domain_id"]
                     ? params["domain_id"].as<int>()
                     : context.env.applicationConfig().robot.domain_id;
    qos_depth_ = params["qos_depth"] ? params["qos_depth"].as<int>() : 1;
    best_effort_ =
        params["best_effort"] ? params["best_effort"].as<bool>() : true;
    anchor_body_name_ = params["anchor_body"]
                            ? params["anchor_body"].as<std::string>()
                            : "base_link";
    if (anchor_body_name_.empty())
      throw std::runtime_error(where + ".params.anchor_body must not be empty");
    const YAML::Node names = params["joint_names"];
    if (!names || !names.IsSequence() || names.size() == 0)
      throw std::runtime_error(where + ".params.joint_names is required");
    for (const auto &name : names)
      joint_names_.push_back(name.as<std::string>());
  }

  bool start() override {
    rclcpp::InitOptions init;
    init.set_domain_id(domain_id_);
    context_ = std::make_shared<rclcpp::Context>();
    context_->init(0, nullptr, init);
    rclcpp::NodeOptions options;
    options.context(context_);
    node_ = rclcpp::Node::make_shared(node_name_, options);
    auto qos = rclcpp::SensorDataQoS().keep_last(qos_depth_);
    qos.reliability(best_effort_ ? RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
                                 : RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    subscription_ = node_->create_subscription<std_msgs::msg::String>(
        topic_, qos, [this](const std_msgs::msg::String &message) {
          try {
            ingest(message.data);
          } catch (const std::exception &error) {
            RCLCPP_ERROR(node_->get_logger(),
                         "invalid motion-reference payload: %s", error.what());
          }
        });
    // The publisher retains the current state; receive it when inference
    // starts after teleop so an already-active session is not a resume edge.
    auto active_qos = rclcpp::QoS(1).reliable().transient_local();
    active_subscription_ = node_->create_subscription<std_msgs::msg::Bool>(
        "/gmr/teleop/active", active_qos,
        [this](const std_msgs::msg::Bool &message) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (message.data == active_ && active_state_known_)
            return;
          if (!message.data) {
            if (!history_.empty()) {
              hold_yaw_ = std::remainder(
                  rootYaw(latest_root_rot_) + yaw_offset_, 2.0 * std::acos(-1.0));
              hold_yaw_valid_ = true;
            }
            pause_seen_ = true;
            resume_pending_ = false;
            frames_until_resume_plus_five_ = 0;
          } else {
            resume_pending_ = pause_seen_ && hold_yaw_valid_;
            pause_seen_ = false;
          }
          active_ = message.data;
          active_state_known_ = true;
        });
    rclcpp::ExecutorOptions executor_options;
    executor_options.context = context_;
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(
        executor_options);
    executor_->add_node(node_);
    thread_ = std::thread([this] { executor_->spin(); });
    return true;
  }

  void stop() override {
    if (executor_)
      executor_->cancel();
    if (thread_.joinable())
      thread_.join();
    executor_.reset();
    active_subscription_.reset();
    subscription_.reset();
    node_.reset();
    if (context_ && context_->is_valid())
      context_->shutdown("motion source stopped");
    context_.reset();
  }

  void reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    // A policy reset also happens on low-level mode re-entry. Clear the
    // command history, but keep the VR heading correction across that reset.
    history_.clear();
    received_at_ = Clock::time_point{};
    frames_until_resume_plus_five_ = 0;
  }

  bool waitUntilReady(std::size_t frame_count,
                      std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mutex_);
    return ready_.wait_for(lock, timeout,
                           [&] { return history_.size() >= frame_count; });
  }

  bool frames(uint64_t, std::size_t frame_count,
              std::vector<MotionReferenceFrame> &output) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (history_.size() < frame_count)
      return false;
    output.assign(history_.end() - static_cast<std::ptrdiff_t>(frame_count),
                  history_.end());
    // Rotate the whole requested window, including frames received before
    // resume, by the same current world yaw.
    const Eigen::Quaterniond yaw_correction(
        Eigen::AngleAxisd(yaw_offset_, Eigen::Vector3d::UnitZ()));
    for (auto &frame : output)
      for (auto &orientation : frame.world_from_body)
        orientation = (yaw_correction * orientation).normalized();
    return true;
  }

  bool stale(Clock::time_point now,
             std::chrono::milliseconds maximum_age) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return history_.empty() || now - received_at_ > maximum_age;
  }

private:
  static double rootYaw(const std::array<double, 4> &root) {
    const Eigen::Quaterniond rotation(root[0], root[1], root[2], root[3]);
    const Eigen::Matrix3d matrix = rotation.normalized().toRotationMatrix();
    return std::atan2(matrix(1, 0), matrix(0, 0));
  }

  static double rootYawRate(const Eigen::Quaterniond &root,
                            const Eigen::Vector3d &world_angular_velocity) {
    const Eigen::Vector3d heading = root * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d heading_rate = world_angular_velocity.cross(heading);
    const double horizontal_norm_squared = heading.x() * heading.x() +
                                           heading.y() * heading.y();
    if (horizontal_norm_squared < 1.0e-8)
      return 0.0;
    return (heading.x() * heading_rate.y() -
            heading.y() * heading_rate.x()) /
           horizontal_norm_squared;
  }

  void logRootYaw(const char *stage, uint64_t sequence,
                  const std::array<double, 4> &root) const {
    const double yaw = rootYaw(root);
    RCLCPP_INFO(node_->get_logger(),
                "motion yaw [%s] seq=%llu yaw=%.6f rad "
                "root_rot(wxyz)=[%.6f, %.6f, %.6f, %.6f]",
                stage, static_cast<unsigned long long>(sequence), yaw,
                root[0], root[1], root[2], root[3]);
  }

  void ingest(const std::string &payload) {
    std::vector<double> q, qd, root, local;
    std::vector<std::string> bodies;
    if (!NumberArray(payload, "dof_pos", q) ||
        !NumberArray(payload, "dof_vel", qd) ||
        !NumberArray(payload, "root_rot", root) || root.size() < 4 ||
        !StringArray(payload, "link_body_list", bodies) ||
        !NumberArray(payload, "keybody_rot_local", local) ||
        q.size() != joint_names_.size() || qd.size() != joint_names_.size() ||
        local.size() != bodies.size() * 4)
      return;
    MotionReferenceFrame frame;
    frame.sequence = ++sequence_;
    frame.joint_names = joint_names_;
    frame.joint_position = Eigen::Map<Eigen::VectorXd>(q.data(), q.size());
    frame.joint_velocity = Eigen::Map<Eigen::VectorXd>(qd.data(), qd.size());
    if (!frame.joint_position.allFinite() || !frame.joint_velocity.allFinite())
      return;
    Eigen::Quaterniond root_rotation(root[0], root[1], root[2], root[3]);
    if (!std::isfinite(root_rotation.norm()) || root_rotation.norm() < 1.0e-12)
      return;
    root_rotation.normalize();
    frame.anchor_body_name = anchor_body_name_;
    frame.body_names = bodies;
    for (std::size_t i = 0; i < frame.body_names.size(); ++i) {
      Eigen::Quaterniond local_rotation(local[i * 4], local[i * 4 + 1],
                                        local[i * 4 + 2], local[i * 4 + 3]);
      if (!std::isfinite(local_rotation.norm()) ||
          local_rotation.norm() < 1.0e-12)
        return;
      frame.world_from_body.push_back(
          (root_rotation * local_rotation.normalized()).normalized());
    }

    std::vector<double> body_position;
    if (NumberArray(payload, "keybody_pos_local", body_position) &&
        body_position.size() == bodies.size() * 3) {
      frame.anchor_from_body_position.reserve(bodies.size());
      for (std::size_t i = 0; i < bodies.size(); ++i)
        frame.anchor_from_body_position.emplace_back(body_position[i * 3],
                                                     body_position[i * 3 + 1],
                                                     body_position[i * 3 + 2]);
    }

    // GMR may omit the anchor from key bodies. Add it explicitly so command
    // configurations can consistently select it by name.
    const auto anchor = std::find(frame.body_names.begin(),
                                  frame.body_names.end(), anchor_body_name_);
    if (anchor == frame.body_names.end()) {
      frame.body_names.insert(frame.body_names.begin(), anchor_body_name_);
      frame.world_from_body.insert(frame.world_from_body.begin(),
                                   root_rotation);
      if (!frame.anchor_from_body_position.empty())
        frame.anchor_from_body_position.insert(
            frame.anchor_from_body_position.begin(), Eigen::Vector3d::Zero());
    } else {
      const std::size_t index = static_cast<std::size_t>(
          std::distance(frame.body_names.begin(), anchor));
      frame.world_from_body[index] = root_rotation;
      if (!frame.anchor_from_body_position.empty())
        frame.anchor_from_body_position[index].setZero();
    }

    std::vector<double> root_position;
    if (NumberArray(payload, "root_pos", root_position) &&
        root_position.size() >= 3) {
      frame.world_anchor_position =
          Eigen::Vector3d(root_position[0], root_position[1], root_position[2]);
      frame.world_anchor_position_valid =
          frame.world_anchor_position.allFinite();
    }
    std::vector<double> root_velocity;
    if (NumberArray(payload, "root_vel", root_velocity) &&
        root_velocity.size() >= 3) {
      frame.anchor_linear_velocity =
          root_rotation.conjugate() *
          Eigen::Vector3d(root_velocity[0], root_velocity[1], root_velocity[2]);
      frame.anchor_linear_velocity_valid =
          frame.anchor_linear_velocity.allFinite();
    }
    std::vector<double> root_angular_velocity;
    Eigen::Vector3d world_angular_velocity = Eigen::Vector3d::Zero();
    if (NumberArray(payload, "root_angvel", root_angular_velocity) &&
        root_angular_velocity.size() >= 3) {
      world_angular_velocity =
          Eigen::Vector3d(root_angular_velocity[0], root_angular_velocity[1],
                          root_angular_velocity[2]);
      frame.anchor_angular_velocity =
          root_rotation.conjugate() * world_angular_velocity;
      frame.anchor_angular_velocity_valid =
          frame.anchor_angular_velocity.allFinite();
    }
    const uint64_t frame_sequence = frame.sequence;
    bool log_initial = false;
    bool log_resume = false;
    bool log_resume_plus_five = false;
    bool log_offset = false;
    double offset_for_log = 0.0;
    double target_for_log = 0.0;
    const std::array<double, 4> root_components{root[0], root[1], root[2],
                                                 root[3]};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const double raw_yaw = rootYaw(root_components);
      if (!hold_yaw_valid_) {
        hold_yaw_ = raw_yaw;
        hold_yaw_valid_ = true;
      }
      const bool hold_heading =
          (active_state_known_ && !active_) || resume_pending_;
      if (hold_heading) {
        yaw_offset_ = std::remainder(hold_yaw_ - raw_yaw,
                                      2.0 * std::acos(-1.0));
        if (frame.anchor_angular_velocity_valid) {
          // The world yaw correction changes with the raw heading while
          // paused. Remove that heading rate from the local root velocity.
          const double yaw_rate = rootYawRate(root_rotation,
                                              world_angular_velocity);
          frame.anchor_angular_velocity -= root_rotation.conjugate() *
              (yaw_rate * Eigen::Vector3d::UnitZ());
        }
      }
      if (!initial_logged_) {
        log_initial = true;
        initial_logged_ = true;
      }
      if (resume_pending_) {
        log_resume = true;
        log_offset = true;
        offset_for_log = yaw_offset_;
        target_for_log = hold_yaw_;
        resume_pending_ = false;
        frames_until_resume_plus_five_ = 5;
      } else if (frames_until_resume_plus_five_ > 0 &&
                 --frames_until_resume_plus_five_ == 0) {
        log_resume_plus_five = true;
      }
      history_.push_back(std::move(frame));
      while (history_.size() > 512)
        history_.pop_front();
      received_at_ = Clock::now();
      latest_root_rot_ = root_components;
    }
    if (log_initial)
      logRootYaw("initial", frame_sequence, root_components);
    if (log_resume)
      logRootYaw("resume", frame_sequence, root_components);
    if (log_offset)
      RCLCPP_INFO(node_->get_logger(),
                  "motion yaw hold target=%.6f rad offset=%.6f rad",
                  target_for_log, offset_for_log);
    if (log_resume_plus_five)
      logRootYaw("resume+5", frame_sequence, root_components);
    ready_.notify_all();
  }

  std::string topic_;
  std::string node_name_;
  int domain_id_ = 0;
  int qos_depth_ = 1;
  bool best_effort_ = true;
  std::string anchor_body_name_ = "base_link";
  std::vector<std::string> joint_names_;
  std::array<double, 4> latest_root_rot_{};
  double hold_yaw_ = 0.0;
  double yaw_offset_ = 0.0;
  bool hold_yaw_valid_ = false;
  bool active_state_known_ = false;
  bool active_ = false;
  bool pause_seen_ = false;
  bool initial_logged_ = false;
  bool resume_pending_ = false;
  int frames_until_resume_plus_five_ = 0;
  std::atomic<uint64_t> sequence_{0};
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<MotionReferenceFrame> history_;
  Clock::time_point received_at_{};
  rclcpp::Context::SharedPtr context_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread thread_;
};

} // namespace

std::unique_ptr<MotionReferenceSource>
createMotionReferenceSource(const YAML::Node &source,
                            const FactoryContext &context,
                            const std::string &where) {
  if (!source || !source.IsMap())
    throw std::runtime_error(where + ".source must be a map");
  const std::string type =
      Required<std::string>(source, "type", where + ".source");
  const YAML::Node params = source["params"];
  if (!params || !params.IsMap())
    throw std::runtime_error(where + ".source.params must be a map");
  if (type == "onnx")
    return std::make_unique<OnnxMotionSource>(params, context,
                                              where + ".source");
  if (type == "ros")
    return std::make_unique<RosMotionSource>(params, context,
                                             where + ".source");
  throw std::runtime_error(where +
                           ": motion-reference commands support source types "
                           "'onnx' and 'ros', got '" +
                           type + "'");
}

} // namespace igris_c_inference::policy::command
