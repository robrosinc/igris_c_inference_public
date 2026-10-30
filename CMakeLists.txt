cmake_minimum_required(VERSION 3.16)
project(igris_c_inference_public)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_link_options("-Wl,--no-as-needed")
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

set(PUBLIC_IGRIS_C_SDK_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/igris_c_sdk/cyclonedds"
    CACHE PATH "IGRIS-C public SDK C++ release prefix")
unset(igris_c_sdk_DIR CACHE)
find_package(
  igris_c_sdk REQUIRED CONFIG PATHS
  "${PUBLIC_IGRIS_C_SDK_PREFIX}/lib/cmake/igris_c_sdk" NO_DEFAULT_PATH)

set(ONNXRUNTIME_ROOT
    "$ENV{CONDA_PREFIX}"
    CACHE PATH "Root of the Pixi-provided ONNX Runtime C++ package")
if(NOT ONNXRUNTIME_ROOT)
  message(
    FATAL_ERROR
      "ONNXRUNTIME_ROOT is empty. Configure this project through the Pixi environment."
  )
endif()

# Never fall back to /usr or /usr/local. The Pixi prefix is the sole ONNX
# Runtime provider for this project.
unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
unset(ONNXRUNTIME_LIBRARY CACHE)
find_path(
  ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_c_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
  PATH_SUFFIXES onnxruntime/core/session onnxruntime
  NO_DEFAULT_PATH)
find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime libonnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(
    FATAL_ERROR
      "ONNX Runtime C++ was not found below ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
  )
endif()
message(STATUS "Using ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(
  onnxruntime::onnxruntime
  PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
             INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

add_library(
  igris_c_inference_core
  src/core/config.cpp
  src/io/robot_io.cpp
  src/modules/policy/RlEnvModule.cpp
  src/modules/policy/managers/ActionManager.cpp
  src/modules/policy/managers/CommandManager.cpp
  src/modules/policy/managers/EventManager.cpp
  src/modules/policy/managers/ObservationManager.cpp
  src/modules/policy/managers/TerminationManager.cpp
  src/modules/policy/mdp/actions/ActionRegistry.cpp
  src/modules/policy/mdp/actions/JointActionTerm.cpp
  src/modules/policy/mdp/actions/JointPositionAction.cpp
  src/modules/policy/mdp/actions/MotionCommandAction.cpp
  src/modules/policy/mdp/actions/SafeJointPositionAction.cpp
  src/modules/policy/mdp/commands/CommandRegistry.cpp
  src/modules/policy/mdp/commands/MotionReferenceCommand.cpp
  src/modules/policy/mdp/commands/SonicMotionReferenceCommand.cpp
  src/modules/policy/mdp/commands/VelocityCommand.cpp
  src/modules/policy/mdp/commands/utils/MotionReferenceSource.cpp
  src/modules/policy/mdp/events/EventRegistry.cpp
  src/modules/policy/mdp/observations/CommandObservations.cpp
  src/modules/policy/mdp/observations/ObservationRegistry.cpp
  src/modules/policy/mdp/observations/PublicObservations.cpp
  src/modules/policy/mdp/observations/RobotStateObservations.cpp
  src/modules/policy/mdp/terminations/TerminationRegistry.cpp
  src/modules/policy/mdp/terminations/TimeOutTermination.cpp
  src/modules/policy/runtime/PolicyRuntime.cpp
  src/modules/policy/runtime/RlEnvConfig.cpp
  src/modules/state_handler.cpp
  src/controller/robotHandler.cpp
  src/utils/joint_alignment.cpp)
target_include_directories(
  igris_c_inference_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(
  igris_c_inference_core
  PUBLIC Eigen3::Eigen igris_c_sdk::igris_c_sdk rclcpp::rclcpp
         ${std_msgs_TARGETS} yaml-cpp::yaml-cpp onnxruntime::onnxruntime)

add_executable(igris_c_inference_node src/main.cpp)
target_link_libraries(igris_c_inference_node PRIVATE igris_c_inference_core)

install(TARGETS igris_c_inference_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY docs/ DESTINATION share/${PROJECT_NAME}/docs)
install(DIRECTORY interfaces/ DESTINATION share/${PROJECT_NAME}/interfaces)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
