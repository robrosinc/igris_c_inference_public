#include "igris_c_inference/modules/policy/mdp/commands/CommandTerm.h"

namespace igris_c_inference::policy::command {

const Registry &registry() {
  static const Registry factories = {
      {"velocity_command", createVelocityCommand},
      {"motion_reference", createMotionReferenceCommand},
      {"sonic_motion_reference", createSonicMotionReferenceCommand},
  };
  return factories;
}

} // namespace igris_c_inference::policy::command
