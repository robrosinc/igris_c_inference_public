#include "igris_c_inference/modules/policy/managers/EventManager.h"

#include "igris_c_inference/modules/policy/RlEnvModule.h"
#include "igris_c_inference/modules/policy/mdp/events/EventTerm.h"
#include "igris_c_inference/modules/policy/utils/YamlConfig.h"

#include <yaml-cpp/yaml.h>

#include <mutex>
#include <stdexcept>
#include <utility>

namespace igris_c_inference::policy {

class EventManager::Impl {
public:
  struct Entry {
    std::string name;
    std::string function;
    std::string mode;
    std::unique_ptr<event::Term> term;
  };

  std::vector<Entry> entries;
  mutable std::mutex mutex;
};

EventManager::EventManager(RlEnvModule &env)
    : ManagerBase(env), impl_(std::make_unique<Impl>()) {}
EventManager::~EventManager() = default;

int EventManager::configure(const std::string &path, std::string &error) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    yaml::checkKeys(root, {"igris_c_inference_events"}, "event root");
    const YAML::Node config = root["igris_c_inference_events"];
    yaml::checkKeys(config, {"terms"}, "events");
    const YAML::Node terms = config["terms"];
    if (!terms || !terms.IsMap())
      throw std::runtime_error("events.terms must be a map");
    impl_->entries.clear();
    const auto &factories = event::registry();
    const event::FactoryContext context{env()};
    for (const auto &entry : terms) {
      const std::string name = entry.first.as<std::string>();
      const YAML::Node node = entry.second;
      yaml::checkKeys(node, {"function", "mode", "params"},
                      "events.terms." + name);
      const std::string function =
          yaml::required<std::string>(node, "function", "events.terms." + name);
      const std::string mode =
          yaml::required<std::string>(node, "mode", "events.terms." + name);
      if (mode != "startup" && mode != "reset")
        throw std::runtime_error("events.terms." + name +
                                 ".mode must be startup or reset");
      const auto factory = factories.find(function);
      if (factory == factories.end())
        throw std::runtime_error("unknown event function: " + function);
      auto term = factory->second(
          name, yaml::optionalMap(node, "params", "events.terms." + name),
          context);
      if (!term)
        throw std::runtime_error("event factory returned null: " + function);
      impl_->entries.push_back(
          Impl::Entry{name, function, mode, std::move(term)});
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

void EventManager::reset() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (auto &entry : impl_->entries)
    entry.term->reset();
}

std::vector<TermDescriptor> EventManager::termDescriptors() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::vector<TermDescriptor> result;
  result.reserve(impl_->entries.size());
  for (const auto &entry : impl_->entries)
    result.push_back(
        TermDescriptor{entry.name, entry.function, {}, entry.mode, 0, {}, {}});
  return result;
}

int EventManager::apply(const std::string &mode,
                        const LifecycleContext &context, std::string &error) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto &entry : impl_->entries) {
      if (entry.mode == mode)
        entry.term->apply(context);
    }
    error.clear();
    return 0;
  } catch (const std::exception &exception) {
    error = exception.what();
    return -1;
  }
}

} // namespace igris_c_inference::policy
