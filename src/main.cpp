#include "igris_c_inference/robotHandler.h"

int main(int argc, char **argv) {
  igris_c_inference::RobotHandler robot(argc, argv);
  const int initialization = robot.initialize();
  if (initialization > 0)
    return 0;
  if (initialization < 0)
    return 1;
  return robot.execute() == 0 ? 0 : 1;
}
