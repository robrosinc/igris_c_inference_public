#include "igris_c_inference/modules/policy/mdp/commands/CommandTerm.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/mdp/commands/utils/MotionReferenceSource.h"
#include "igris_c_inference/utils/joint_alignment.hpp"
#include "igris_c_inference/utils/reference_heading_alignment.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>

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

Eigen::Matrix3d worldFromFrame(const ProcessedState &state,
                               const RobotFrameConfig &frame) {
  Eigen::Matrix3d rotation = state.world_from_base;
  for (const auto &joint : frame.joints) {
    const double q = state.joint_position(joint.robot_index);
    rotation *= Eigen::AngleAxisd(q, joint.axis).toRotationMatrix();
  }
  return rotation;
}

class SonicMotionReferenceCommand final : public Term {
public:
  SonicMotionReferenceCommand(std::string name, const YAML::Node &config,
                              const FactoryContext &context)
      : Term(std::move(name), context.env), robot_(context.env.robotConfig()) {
    const std::string where = "commands.terms." + this->name();
    const YAML::Node params = config["params"];
    if (!params || !params.IsMap())
      throw std::runtime_error(where + ".params must be a map");
    alignment_.configure(
        ResolvePath(context, Required<std::string>(params, "alignment",
                                                   where + ".params")),
        robot_);
    robot_frame_ = robot_.frame(
        Required<std::string>(params, "robot_frame", where + ".params"));
    reference_body_ =
        Required<std::string>(params, "reference_body", where + ".params");
    heading_alignment_.configure(
        params["heading_alignment"]
            ? params["heading_alignment"].as<std::string>()
            : "raw",
        where + ".params.heading_alignment");
    history_length_ =
        Required<std::size_t>(params, "history_length", where + ".params");
    history_stride_ =
        Required<std::size_t>(params, "history_stride", where + ".params");
    if (history_length_ == 0 || history_stride_ == 0)
      throw std::runtime_error(where +
                               ": history length and stride must be positive");
    history_capacity_ = 1 + (history_length_ - 1) * history_stride_;
    const std::string order = params["history_order"]
                                  ? params["history_order"].as<std::string>()
                                  : "oldest_to_newest";
    if (order == "oldest_to_newest")
      oldest_first_ = true;
    else if (order == "newest_to_oldest")
      oldest_first_ = false;
    else
      throw std::runtime_error(where +
                               ": history_order must be oldest_to_newest or "
                               "newest_to_oldest");
    const int stale_ms =
        params["stale_after_ms"] ? params["stale_after_ms"].as<int>() : 100;
    if (stale_ms <= 0)
      throw std::runtime_error(where +
                               ".params.stale_after_ms must be positive");
    stale_after_ = std::chrono::milliseconds(stale_ms);
    const std::string onStale =
        params["on_stale"] ? params["on_stale"].as<std::string>() : "reject";
    if (onStale == "reject")
      stale_behavior_ = StaleBehavior::Reject;
    else if (onStale == "hold_last")
      stale_behavior_ = StaleBehavior::HoldLast;
    else
      throw std::runtime_error(where +
                               ".params.on_stale must be reject or hold_last");
    source_ = createMotionReferenceSource(config["source"], context, where);
  }

  bool start() override { return source_->start(); }
  void stop() override { source_->stop(); }
  void reset() override {
    source_->reset();
    heading_alignment_.reset();
  }
  void resetHeadingAlignment() override { heading_alignment_.reset(); }

  bool waitUntilReady(std::chrono::milliseconds timeout) override {
    return source_->waitUntilReady(history_capacity_, timeout);
  }

  SignalMap prepare(const StepContext &step,
                    const PrepareContext &context) override {
    if (!step.status.valid)
      throw std::runtime_error("command '" + name() +
                               "' requires valid robot state");
    if (source_->stale(step.now, stale_after_) &&
        stale_behavior_ == StaleBehavior::Reject)
      throw std::runtime_error("command '" + name() +
                               "' motion source is stale");
    const Eigen::Matrix3d world_from_robot_frame =
        worldFromFrame(step.status, robot_frame_);
    if (heading_alignment_.requiresLatch()) {
      std::vector<MotionReferenceFrame> current;
      if (!source_->frames(context.policyStep, 1, current) ||
          current.size() != 1)
        throw std::runtime_error("command '" + name() +
                                 "' lacks a current motion frame");
      heading_alignment_.relative(
          world_from_robot_frame,
          referenceRotation(current.front()).toRotationMatrix());
    }
    std::vector<MotionReferenceFrame> history;
    if (!source_->frames(context.policyStep, history_capacity_, history) ||
        history.size() != history_capacity_)
      throw std::runtime_error("command '" + name() +
                               "' lacks the configured motion history");

    std::vector<const MotionReferenceFrame *> selected;
    selected.reserve(history_length_);
    for (std::size_t i = 0; i < history_length_; ++i) {
      const std::size_t chronological =
          oldest_first_ ? i : history_length_ - 1 - i;
      const std::size_t index =
          (history_length_ - 1 - chronological) * history_stride_;
      selected.push_back(&history[history.size() - 1 - index]);
    }

    const Eigen::Index source_dofs =
        static_cast<Eigen::Index>(alignment_.sourceDofs());
    const Eigen::Index samples = static_cast<Eigen::Index>(history_length_);
    Eigen::VectorXd output(samples * source_dofs * 2 + samples * 6);
    // The released encoder contract stores every q frame first, followed by
    // every qdot frame; it is not interleaved per timestep.
    Eigen::Index q_offset = 0;
    Eigen::Index qd_offset = samples * source_dofs;
    Eigen::Index orientation_offset = samples * source_dofs * 2;
    for (const MotionReferenceFrame *frame : selected) {
      const Eigen::VectorXd q = alignment_.sourcePosition(
          frame->joint_position, frame->joint_names, false);
      const Eigen::VectorXd qd =
          alignment_.sourceVelocity(frame->joint_velocity, frame->joint_names);
      output.segment(q_offset, source_dofs) = q;
      output.segment(qd_offset, source_dofs) = qd;
      q_offset += source_dofs;
      qd_offset += source_dofs;

      const Eigen::Matrix3d robot_from_reference = heading_alignment_.relative(
          world_from_robot_frame, referenceRotation(*frame).toRotationMatrix());
      // SONIC flattens the first two rotation-matrix columns row-major.
      output.segment<6>(orientation_offset) << robot_from_reference(0, 0),
          robot_from_reference(0, 1), robot_from_reference(1, 0),
          robot_from_reference(1, 1), robot_from_reference(2, 0),
          robot_from_reference(2, 1);
      orientation_offset += 6;
    }
    return {{name(), PolicySignal{std::move(output), {}}}};
  }

  SignalMap resetSignals() const override {
    return {{name(), PolicySignal{Eigen::VectorXd::Zero(outputSize()), {}}}};
  }

  std::vector<SignalDescriptor> signalDescriptors() const override {
    return {{name(), static_cast<int>(outputSize())}};
  }

private:
  const Eigen::Quaterniond &
  referenceRotation(const MotionReferenceFrame &frame) const {
    const auto body = std::find(frame.body_names.begin(),
                                frame.body_names.end(), reference_body_);
    if (body == frame.body_names.end())
      throw std::runtime_error("motion frame is missing reference body '" +
                               reference_body_ + "'");
    const std::size_t body_index =
        static_cast<std::size_t>(std::distance(frame.body_names.begin(), body));
    if (body_index >= frame.world_from_body.size())
      throw std::runtime_error("motion body quaternion layout is invalid");
    return frame.world_from_body[body_index];
  }

  Eigen::Index outputSize() const {
    const Eigen::Index samples = static_cast<Eigen::Index>(history_length_);
    const Eigen::Index sourceDofs =
        static_cast<Eigen::Index>(alignment_.sourceDofs());
    return samples * sourceDofs * 2 + samples * 6;
  }

  enum class StaleBehavior { Reject, HoldLast };

  RobotConfig robot_;
  RobotFrameConfig robot_frame_;
  std::string reference_body_;
  JointAlignment alignment_;
  ReferenceHeadingAlignment heading_alignment_;
  std::unique_ptr<MotionReferenceSource> source_;
  std::size_t history_length_ = 10;
  std::size_t history_stride_ = 5;
  std::size_t history_capacity_ = 46;
  bool oldest_first_ = true;
  std::chrono::milliseconds stale_after_{100};
  StaleBehavior stale_behavior_{StaleBehavior::HoldLast};
};

} // namespace

std::unique_ptr<Term>
createSonicMotionReferenceCommand(const std::string &name,
                                  const YAML::Node &config,
                                  const FactoryContext &context) {
  return std::make_unique<SonicMotionReferenceCommand>(name, config, context);
}

} // namespace igris_c_inference::policy::command
