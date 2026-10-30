#include "igris_c_inference/modules/policy/mdp/terminations/TerminationTerm.h"

namespace igris_c_inference::policy::termination {

const Registry &registry() {
  static const Registry factories{
      {"time_out", createTimeOutTermination},
  };
  return factories;
}

} // namespace igris_c_inference::policy::termination
