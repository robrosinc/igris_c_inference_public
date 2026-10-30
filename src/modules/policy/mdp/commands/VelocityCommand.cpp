#include "igris_c_inference/modules/policy/mdp/commands/CommandTerm.h"

#include "igris_c_sdk/igris_c_msgs.hpp"
#include "igris_c_sdk/qos.hpp"
#include "igris_c_sdk/subscriber.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
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

class VelocityCommand final : public Term {
public:
  VelocityCommand(std::string name, RlEnvModule &env, const YAML::Node &config)
      : Term(std::move(name), env) {
    const std::string where = "commands.terms." + this->name();
    const YAML::Node source = config["source"];
    if (!source || !source.IsMap())
      throw std::runtime_error(where + ".source must be a map");
    const std::string source_type =
        Required<std::string>(source, "type", where + ".source");
    if (source_type != "dds")
      throw std::runtime_error(where +
                               ": velocity_command supports only source type "
                               "'dds', got '" +
                               source_type + "'");
    const YAML::Node source_params = source["params"];
    topic_ =
        Required<std::string>(source_params, "topic", where + ".source.params");
    if (topic_.empty())
      throw std::runtime_error(where + ".source.params.topic is empty");

    const YAML::Node params = config["params"];
    if (!params || !params.IsMap())
      throw std::runtime_error(where + ".params must be a map");
    const int stale_ms =
        Required<int>(params, "stale_after_ms", where + ".params");
    if (stale_ms <= 0)
      throw std::runtime_error(where +
                               ".params.stale_after_ms must be positive");
    stale_after_ = std::chrono::milliseconds(stale_ms);
    const std::string stale =
        Required<std::string>(params, "on_stale", where + ".params");
    if (stale == "reject")
      stale_behavior_ = StaleBehavior::Reject;
    else if (stale == "zero")
      stale_behavior_ = StaleBehavior::Zero;
    else if (stale == "hold_last")
      stale_behavior_ = StaleBehavior::HoldLast;
    else
      throw std::runtime_error(where +
                               ".params.on_stale must be reject, zero, or "
                               "hold_last");

    const YAML::Node components = params["components"];
    if (!components || !components.IsSequence() || components.size() == 0)
      throw std::runtime_error(
          where + ".params.components must be a non-empty sequence");
    const std::unordered_set<std::string> supported = {
        "linear_x",  "linear_y",  "linear_z",
        "angular_x", "angular_y", "angular_z"};
    std::unordered_set<std::string> seen;
    for (const YAML::Node &node : components) {
      Axis axis;
      axis.field =
          Required<std::string>(node, "field", where + ".params.components");
      axis.scale =
          Required<double>(node, "scale", where + ".params.components");
      axis.minimum =
          Required<double>(node, "min", where + ".params.components");
      axis.maximum =
          Required<double>(node, "max", where + ".params.components");
      if (supported.find(axis.field) == supported.end())
        throw std::runtime_error(where + ": unsupported TwistStamped field '" +
                                 axis.field + "'");
      if (!seen.emplace(axis.field).second)
        throw std::runtime_error(where + ": duplicate TwistStamped field '" +
                                 axis.field + "'");
      if (!std::isfinite(axis.scale) || !std::isfinite(axis.minimum) ||
          !std::isfinite(axis.maximum) || axis.minimum >= axis.maximum)
        throw std::runtime_error(where + ": invalid component scale or limits");
      axes_.push_back(std::move(axis));
    }
    latest_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(axes_.size()));
  }

  bool start() override {
    subscriber_ = std::make_unique<
        igris_c_sdk::Subscriber<geometry_msgs::msg::dds_::TwistStamped_>>(
        topic_, igris_c_sdk::QosProfile::Default());
    return subscriber_->init(
        [this](const auto &message) { ingest(message, Clock::now()); });
  }

  void stop() override { subscriber_.reset(); }

  void reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_.setZero();
    received_at_ = Clock::time_point{};
    has_sample_ = false;
  }

  bool waitUntilReady(std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mutex_);
    return sample_cv_.wait_for(lock, timeout, [this] { return has_sample_; });
  }

  SignalMap prepare(const StepContext &step, const PrepareContext &) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_sample_)
      throw std::runtime_error("command '" + name() + "' has no DDS sample");
    const bool stale = step.now - received_at_ > stale_after_;
    if (stale && stale_behavior_ == StaleBehavior::Reject)
      throw std::runtime_error("command '" + name() + "' is stale");
    Eigen::VectorXd value;
    if (stale && stale_behavior_ == StaleBehavior::Zero)
      value = Eigen::VectorXd::Zero(latest_.size());
    else
      value = latest_;
    return {{name(), PolicySignal{std::move(value), {}}}};
  }

  SignalMap resetSignals() const override {
    return {{name(), PolicySignal{Eigen::VectorXd::Zero(
                                      static_cast<Eigen::Index>(axes_.size())),
                                  {}}}};
  }

  std::vector<SignalDescriptor> signalDescriptors() const override {
    return {{name(), static_cast<int>(axes_.size())}};
  }

  void ingest(const geometry_msgs::msg::dds_::TwistStamped_ &message,
              Clock::time_point received_at) {
    Eigen::VectorXd values(static_cast<Eigen::Index>(axes_.size()));
    for (std::size_t i = 0; i < axes_.size(); ++i) {
      const Axis &axis = axes_[i];
      values(static_cast<Eigen::Index>(i)) =
          std::clamp(readField(message, axis.field) * axis.scale, axis.minimum,
                     axis.maximum);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_ = std::move(values);
      received_at_ = received_at;
      has_sample_ = true;
    }
    sample_cv_.notify_all();
  }

private:
  enum class StaleBehavior { Reject, Zero, HoldLast };

  struct Axis {
    std::string field;
    double scale = 1.0;
    double minimum = 0.0;
    double maximum = 0.0;
  };

  static double
  readField(const geometry_msgs::msg::dds_::TwistStamped_ &message,
            const std::string &field) {
    if (field == "linear_x")
      return message.twist().linear().x();
    if (field == "linear_y")
      return message.twist().linear().y();
    if (field == "linear_z")
      return message.twist().linear().z();
    if (field == "angular_x")
      return message.twist().angular().x();
    if (field == "angular_y")
      return message.twist().angular().y();
    if (field == "angular_z")
      return message.twist().angular().z();
    throw std::runtime_error("unsupported TwistStamped field: " + field);
  }

  std::string topic_;
  std::chrono::milliseconds stale_after_{200};
  StaleBehavior stale_behavior_ = StaleBehavior::Reject;
  std::vector<Axis> axes_;

  std::mutex mutex_;
  std::condition_variable sample_cv_;
  Eigen::VectorXd latest_;
  Clock::time_point received_at_{};
  bool has_sample_ = false;
  std::unique_ptr<
      igris_c_sdk::Subscriber<geometry_msgs::msg::dds_::TwistStamped_>>
      subscriber_;
};

} // namespace

std::unique_ptr<Term> createVelocityCommand(const std::string &name,
                                            const YAML::Node &config,
                                            const FactoryContext &context) {
  return std::make_unique<VelocityCommand>(name, context.env, config);
}

} // namespace igris_c_inference::policy::command
