#include "igris_c_inference/modules/policy/mdp/events/EventTerm.h"

namespace igris_c_inference::policy::event {

const Registry &registry() {
  static const Registry factories{};
  return factories;
}

} // namespace igris_c_inference::policy::event
