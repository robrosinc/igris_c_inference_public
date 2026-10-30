#include "igris_c_inference/modules/policy/mdp/commands/CommandTerm.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/mdp/commands/utils/MotionReferenceSource.h"
#include "igris_c_inference/utils/reference_heading_alignment.hpp"

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace igris_c_inference::policy::command {
namespace {

template <typename T>
T Required(const YAML::Node &node, const char *key, const std::string &where) {
  if (!node || !node[key])
    throw std::runtime_error(where + ": missing '" + key + "'");
  return node[key].as<T>();
}

void CheckKeys(const YAML::Node &node,
               std::initializer_list<std::string_view> allowed,
               const std::string &where) {
  if (!node || !node.IsMap())
    throw std::runtime_error(where + " must be a map");
  std::unordered_set<std::string> valid;
  for (const std::string_view key : allowed)
    valid.emplace(key);
  for (const auto &entry : node) {
    const std::string key = entry.first.as<std::string>();
    if (valid.find(key) == valid.end())
      throw std::runtime_error(where + ": unknown key '" + key + "'");
  }
}

std::vector<std::string> ReadNames(const YAML::Node &node,
                                   const std::string &where) {
  if (!node || !node.IsSequence() || node.size() == 0)
    throw std::runtime_error(where + " must be a non-empty sequence");
  std::vector<std::string> names;
  std::unordered_set<std::string> unique;
  names.reserve(node.size());
  for (const auto &entry : node) {
    const std::string name = entry.as<std::string>();
    if (name.empty() || !unique.emplace(name).second)
      throw std::runtime_error(where + " must contain unique non-empty names");
    names.push_back(name);
  }
  return names;
}

std::unordered_map<std::string, std::size_t>
IndexNames(const std::vector<std::string> &names, const std::string &label) {
  std::unordered_map<std::string, std::size_t> indices;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty() || !indices.emplace(names[i], i).second)
      throw std::runtime_error("motion frame has invalid " + label + " names");
  }
  return indices;
}

enum class Field {
  JointPosition,
  JointVelocity,
  BodyPosition,
  BodyOrientation,
  AnchorLinearVelocity,
  AnchorAngularVelocity,
  AnchorOrientation,
  AnchorHeight,
};

struct Component {
  Field field = Field::JointPosition;
  std::vector<std::string> names;
  bool use_named_robot_frame = false;
  RobotFrameConfig robot_frame;
};

Eigen::Matrix3d worldFromFrame(const ProcessedState &state,
                               const RobotFrameConfig &frame) {
  Eigen::Matrix3d rotation = state.world_from_base;
  for (const auto &joint : frame.joints) {
    const double q = state.joint_position(joint.robot_index);
    rotation *= Eigen::AngleAxisd(q, joint.axis).toRotationMatrix();
  }
  return rotation;
}

class MotionReferenceCommand final : public Term {
public:
  MotionReferenceCommand(std::string name, const YAML::Node &config,
                         const FactoryContext &context)
      : Term(std::move(name), context.env) {
    const std::string where = "commands.terms." + this->name();
    const YAML::Node params = config["params"];
    CheckKeys(params,
              {"anchor_body", "heading_alignment", "stale_after_ms", "on_stale",
               "history_length", "history_stride", "history_order",
               "components"},
              where + ".params");
    anchor_body_ = params["anchor_body"]
                       ? params["anchor_body"].as<std::string>()
                       : "base_link";
    if (anchor_body_.empty())
      throw std::runtime_error(where + ".params.anchor_body must not be empty");
    heading_alignment_.configure(
        params["heading_alignment"]
            ? params["heading_alignment"].as<std::string>()
            : "raw",
        where + ".params.heading_alignment");

    // With the default one-frame window, observation terms may own history.
    // Their stored orientations remain relative to the robot pose at each
    // observation update. Command-owned history instead evaluates every
    // selected source frame with the current StepContext, matching SONIC's
    // current-robot-frame orientation convention.
    history_length_ = params["history_length"]
                          ? params["history_length"].as<std::size_t>()
                          : 1U;
    history_stride_ = params["history_stride"]
                          ? params["history_stride"].as<std::size_t>()
                          : 1U;
    if (history_length_ == 0U || history_stride_ == 0U)
      throw std::runtime_error(where +
                               ": history length and stride must be positive");
    history_capacity_ = 1U + (history_length_ - 1U) * history_stride_;
    const std::string history_order =
        params["history_order"] ? params["history_order"].as<std::string>()
                                : "oldest_to_newest";
    if (history_order == "oldest_to_newest")
      oldest_first_ = true;
    else if (history_order == "newest_to_oldest")
      oldest_first_ = false;
    else
      throw std::runtime_error(
          where +
          ": history_order must be oldest_to_newest or newest_to_oldest");

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

    const YAML::Node components = params["components"];
    if (!components || !components.IsSequence() || components.size() == 0)
      throw std::runtime_error(where + ".params.components must be non-empty");
    for (std::size_t i = 0; i < components.size(); ++i) {
      const YAML::Node node = components[i];
      const std::string component_where =
          where + ".params.components[" + std::to_string(i) + "]";
      CheckKeys(node, {"field", "names", "robot_frame"}, component_where);
      const std::string field =
          Required<std::string>(node, "field", component_where);
      Component component;
      if (field == "joint_position") {
        component.field = Field::JointPosition;
        component.names = ReadNames(node["names"], component_where + ".names");
      } else if (field == "joint_velocity") {
        component.field = Field::JointVelocity;
        component.names = ReadNames(node["names"], component_where + ".names");
      } else if (field == "body_position") {
        component.field = Field::BodyPosition;
        component.names = ReadNames(node["names"], component_where + ".names");
      } else if (field == "body_orientation") {
        component.field = Field::BodyOrientation;
        component.names = ReadNames(node["names"], component_where + ".names");
      } else if (field == "anchor_linear_velocity") {
        component.field = Field::AnchorLinearVelocity;
      } else if (field == "anchor_angular_velocity") {
        component.field = Field::AnchorAngularVelocity;
      } else if (field == "anchor_orientation") {
        component.field = Field::AnchorOrientation;
        const std::string robot_frame =
            node["robot_frame"] ? node["robot_frame"].as<std::string>()
                                : "base_link";
        if (robot_frame != "base_link") {
          component.use_named_robot_frame = true;
          component.robot_frame = context.env.robotConfig().frame(robot_frame);
        }
      } else if (field == "anchor_height") {
        component.field = Field::AnchorHeight;
      } else {
        throw std::runtime_error(component_where + ": unknown field '" + field +
                                 "'");
      }
      if (!component.names.empty() && node["robot_frame"])
        throw std::runtime_error(component_where +
                                 ": robot_frame is only valid for "
                                 "anchor_orientation");
      if (component.names.empty() && node["names"])
        throw std::runtime_error(component_where +
                                 ": names is not valid for this field");
      components_.push_back(std::move(component));
    }
    if (heading_alignment_.requiresLatch()) {
      const auto orientation_count =
          std::count_if(components_.begin(), components_.end(),
                        [](const Component &component) {
                          return component.field == Field::AnchorOrientation;
                        });
      if (orientation_count != 1)
        throw std::runtime_error(
            where + ".params.heading_alignment requires exactly one "
                    "anchor_orientation component");
    }
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

    const auto latch_heading = [this,
                                &step](const MotionReferenceFrame &current) {
      const auto orientation =
          std::find_if(components_.begin(), components_.end(),
                       [](const Component &component) {
                         return component.field == Field::AnchorOrientation;
                       });
      if (orientation == components_.end())
        throw std::runtime_error("command '" + name() +
                                 "' lacks an orientation component");
      const auto joint_indices = IndexNames(current.joint_names, "joint");
      const auto body_indices = IndexNames(current.body_names, "body");
      componentValue(*orientation, current, joint_indices, body_indices,
                     step.status);
    };

    // Multi-frame commands latch against the true current frame before asking
    // an ONNX source for a forward window or a ROS source for recent history.
    if (history_capacity_ > 1U && heading_alignment_.requiresLatch()) {
      std::vector<MotionReferenceFrame> current;
      if (!source_->frames(context.policyStep, 1, current) ||
          current.size() != 1U)
        throw std::runtime_error("command '" + name() +
                                 "' lacks a current motion frame");
      latch_heading(current.front());
    }

    std::vector<MotionReferenceFrame> history;
    if (!source_->frames(context.policyStep, history_capacity_, history) ||
        history.size() != history_capacity_)
      throw std::runtime_error("command '" + name() +
                               "' lacks the configured motion history");
    if (heading_alignment_.requiresLatch())
      latch_heading(history.front());

    std::vector<const MotionReferenceFrame *> selected;
    selected.reserve(history_length_);
    for (std::size_t index = 0; index < history_length_; ++index) {
      const std::size_t chronological =
          oldest_first_ ? index : history_length_ - 1U - index;
      const std::size_t history_index =
          (history_length_ - 1U - chronological) * history_stride_;
      selected.push_back(&history[history.size() - 1U - history_index]);
    }

    Eigen::VectorXd output(outputSize());
    Eigen::Index offset = 0;
    // Components are outermost so history is field-major: every sample for
    // one component is contiguous before the next component begins.
    for (const Component &component : components_) {
      for (const MotionReferenceFrame *frame : selected) {
        const auto joint_indices = IndexNames(frame->joint_names, "joint");
        const auto body_indices = IndexNames(frame->body_names, "body");
        const Eigen::VectorXd value = componentValue(
            component, *frame, joint_indices, body_indices, step.status);
        output.segment(offset, value.size()) = value;
        offset += value.size();
      }
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
  Eigen::VectorXd componentValue(
      const Component &component, const MotionReferenceFrame &frame,
      const std::unordered_map<std::string, std::size_t> &joint_indices,
      const std::unordered_map<std::string, std::size_t> &body_indices,
      const ProcessedState &state) {
    if (component.field == Field::JointPosition ||
        component.field == Field::JointVelocity) {
      const Eigen::VectorXd &source = component.field == Field::JointPosition
                                          ? frame.joint_position
                                          : frame.joint_velocity;
      if (source.size() != static_cast<Eigen::Index>(frame.joint_names.size()))
        throw std::runtime_error("motion frame joint vector size is invalid");
      Eigen::VectorXd output(static_cast<Eigen::Index>(component.names.size()));
      for (std::size_t i = 0; i < component.names.size(); ++i) {
        const auto found = joint_indices.find(component.names[i]);
        if (found == joint_indices.end())
          throw std::runtime_error("motion frame is missing joint '" +
                                   component.names[i] + "'");
        output(static_cast<Eigen::Index>(i)) =
            source(static_cast<Eigen::Index>(found->second));
      }
      return output;
    }
    if (component.field == Field::BodyPosition) {
      if (frame.anchor_from_body_position.size() != frame.body_names.size())
        throw std::runtime_error(
            "motion frame does not provide anchor-frame body positions");
      Eigen::VectorXd output(
          static_cast<Eigen::Index>(component.names.size() * 3));
      for (std::size_t i = 0; i < component.names.size(); ++i) {
        const auto found = body_indices.find(component.names[i]);
        if (found == body_indices.end())
          throw std::runtime_error("motion frame is missing body '" +
                                   component.names[i] + "'");
        output.segment<3>(static_cast<Eigen::Index>(i * 3)) =
            frame.anchor_from_body_position[found->second];
      }
      return output;
    }
    if (component.field == Field::BodyOrientation) {
      requireAnchor(frame);
      if (frame.world_from_body.size() != frame.body_names.size())
        throw std::runtime_error(
            "motion frame does not provide world-frame body orientations");
      const auto anchor = body_indices.find(anchor_body_);
      if (anchor == body_indices.end())
        throw std::runtime_error("motion frame is missing anchor body '" +
                                 anchor_body_ + "'");
      const Eigen::Matrix3d anchor_from_world =
          frame.world_from_body[anchor->second].toRotationMatrix().transpose();
      Eigen::VectorXd output(
          static_cast<Eigen::Index>(component.names.size() * 6));
      for (std::size_t i = 0; i < component.names.size(); ++i) {
        const auto found = body_indices.find(component.names[i]);
        if (found == body_indices.end())
          throw std::runtime_error("motion frame is missing body '" +
                                   component.names[i] + "'");
        const Eigen::Matrix3d anchor_from_body =
            anchor_from_world *
            frame.world_from_body[found->second].toRotationMatrix();
        output.segment<6>(static_cast<Eigen::Index>(i * 6))
            << anchor_from_body(0, 0),
            anchor_from_body(0, 1), anchor_from_body(1, 0),
            anchor_from_body(1, 1), anchor_from_body(2, 0),
            anchor_from_body(2, 1);
      }
      return output;
    }

    requireAnchor(frame);
    if (component.field == Field::AnchorLinearVelocity) {
      if (!frame.anchor_linear_velocity_valid)
        throw std::runtime_error(
            "motion frame does not provide anchor linear velocity");
      return Eigen::VectorXd(frame.anchor_linear_velocity);
    }
    if (component.field == Field::AnchorAngularVelocity) {
      if (!frame.anchor_angular_velocity_valid)
        throw std::runtime_error(
            "motion frame does not provide anchor angular velocity");
      return Eigen::VectorXd(frame.anchor_angular_velocity);
    }
    if (component.field == Field::AnchorHeight) {
      if (!frame.world_anchor_position_valid)
        throw std::runtime_error(
            "motion frame does not provide anchor world position");
      Eigen::VectorXd output(1);
      output(0) = frame.world_anchor_position.z();
      return output;
    }
    const auto body = body_indices.find(anchor_body_);
    if (body == body_indices.end() ||
        body->second >= frame.world_from_body.size())
      throw std::runtime_error(
          "motion frame is missing anchor orientation for '" + anchor_body_ +
          "'");
    const Eigen::Matrix3d world_from_robot =
        component.use_named_robot_frame
            ? worldFromFrame(state, component.robot_frame)
            : state.world_from_base;
    const Eigen::Matrix3d robot_from_anchor = heading_alignment_.relative(
        world_from_robot,
        frame.world_from_body[body->second].toRotationMatrix());
    Eigen::VectorXd output(6);
    output << robot_from_anchor(0, 0), robot_from_anchor(0, 1),
        robot_from_anchor(1, 0), robot_from_anchor(1, 1),
        robot_from_anchor(2, 0), robot_from_anchor(2, 1);
    return output;
  }

  void requireAnchor(const MotionReferenceFrame &frame) const {
    if (frame.anchor_body_name != anchor_body_)
      throw std::runtime_error(
          "motion frame anchor body '" + frame.anchor_body_name +
          "' does not match configured anchor body '" + anchor_body_ + "'");
  }

  Eigen::Index outputSize() const {
    Eigen::Index size = 0;
    for (const Component &component : components_) {
      switch (component.field) {
      case Field::JointPosition:
      case Field::JointVelocity:
        size += static_cast<Eigen::Index>(component.names.size());
        break;
      case Field::BodyPosition:
        size += static_cast<Eigen::Index>(component.names.size() * 3U);
        break;
      case Field::BodyOrientation:
        size += static_cast<Eigen::Index>(component.names.size() * 6U);
        break;
      case Field::AnchorLinearVelocity:
      case Field::AnchorAngularVelocity:
        size += 3;
        break;
      case Field::AnchorOrientation:
        size += 6;
        break;
      case Field::AnchorHeight:
        size += 1;
        break;
      }
    }
    return size * static_cast<Eigen::Index>(history_length_);
  }

  enum class StaleBehavior { Reject, HoldLast };

  std::string anchor_body_;
  std::vector<Component> components_;
  std::unique_ptr<MotionReferenceSource> source_;
  ReferenceHeadingAlignment heading_alignment_;
  std::size_t history_length_{1U};
  std::size_t history_stride_{1U};
  std::size_t history_capacity_{1U};
  bool oldest_first_{true};
  std::chrono::milliseconds stale_after_{100};
  StaleBehavior stale_behavior_{StaleBehavior::HoldLast};
};

} // namespace

std::unique_ptr<Term>
createMotionReferenceCommand(const std::string &name, const YAML::Node &config,
                             const FactoryContext &context) {
  return std::make_unique<MotionReferenceCommand>(name, config, context);
}

} // namespace igris_c_inference::policy::command
