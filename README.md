# IGRIS-C inference

`igris_c_inference_public` runs configurable ONNX policies on IGRIS-C. A policy
directory selects robot state processing, named command terms, observation
terms, action processing, and the model itself. This release bundles the
`gmt_v1.0.0` motion-tracking policy. The runtime also supports other policy
configurations with different command sources, ONNX inputs, and joint subsets.

The bundled policy uses a 300 Hz control path and a 50 Hz policy path:

```text
LowState ──> state processing ──> observation terms ──> ONNX ──> action term
                    ^                    ^                          │
                    │                    │                          v
             named robot frames    command manager              LowCmd
                                      ^
                              DDS / ROS / ONNX source
```

Command-source support belongs to each command term. The built-in
`velocity_command` accepts DDS `WalkTwist`; `motion_reference` and
`sonic_motion_reference` accept a ROS retarget stream or recorded motion ONNX.

## Safety

The node starts in `WAIT` and waits without a startup deadline for a valid
LowState and RobotState. It then enters `IDLE` without publishing LowCmd. Once
the command source is ready, `startup.enter_low_level_mode: true` requests
low-level control once; with `false`, another component must enter that mode.
Only a confirmed low-level RobotState starts `RUNNING` and LowCmd publishing.
The first policy step uses a LowState received after that mode entry.
Leaving low-level mode returns the node to `IDLE` and discards the previous
command. A stale LowState during `RUNNING` enters `ERROR`; the node checks for a
fresh, valid LowState every 20 ms and then returns to `IDLE`. Another low-level
mode exit and entry is required before running again. An invalid state,
inference failure, or publish failure remains a latched `ERROR`. State changes
are printed in red. `SIGINT` or `SIGTERM` exits the process.

`loop.state_timeout_ms` defaults to 100 ms and guards LowState freshness only
while running. `startup.first_state_timeout_ms` and
`startup.command_timeout_ms` remain accepted for existing configurations but
do not end the initial wait. The node does not automatically request low-level
mode again after leaving it; a later external mode transition can resume it.
The bundled policy sets `startup.enter_low_level_mode: false`, so the robot
controller or teleop operator must enter low-level mode.

Test a new policy in simulation, mechanically support the robot, keep the
emergency stop available, and verify joint names, scales, defaults, and output
ranges before enabling motors.

Disable the controller's gravity compensation in the IGRIS-C system
configuration, normally `/robros/configs/robot/robot.yaml`:

```yaml
igris_c:
  controller:
    GravityCompensationModule:
      enabled: false
```

This is the robot controller file, not the policy directory's `robot.yaml`.
Restart IGRIS-C through the normal robot procedure after changing it.

## Build and run

This repository owns an isolated Pixi environment. Install Git and Pixi first;
Go Task is optional. ROS 2 Jazzy, ONNX Runtime, OpenSSL, Eigen, and yaml-cpp
come from that environment instead of `/usr` or
`/usr/local`. The `igris_c_sdk` submodule points to the public SDK
[`v2.1.1`](https://github.com/robrosinc/igris_c_sdk_public/tree/v2.1.1) release.
Its C++ static library and bundled CycloneDDS archives are linked directly from
`thirdparty/igris_c_sdk/cyclonedds`. Build on Ubuntu 24.04 or a host with
glibc 2.39 or newer; the Pixi toolchain uses a matching 2.39 sysroot.

After cloning, initialize the SDK submodule:

```bash
git submodule update --init
```

Build and perform offline dependency validation from this repository's root:

```bash
task build
```

The task runs the following commands. They remain available when Go Task is not
installed:

```bash
pixi install --locked
pixi run --locked build
pixi run --locked verify
```

The build checks that the SDK checkout and recorded submodule commit match the
tag in [`sdk-version.txt`](sdk-version.txt). It does not write to `/usr` or
`/usr/local`. The node is installed under `install/`. To run the node after
the required human safety review:

```bash
pixi run --locked run \
  --config "$PWD/config/gmt_v1.0.0/params.yaml"
```

Run this command from the repository root. The bundled `params.yaml` uses DDS
domain `0`, robot namespace `igris_c_f90b76f7`, and a ROS motion-reference
source on domain `0`. Set these values for your robot and producer before use.
The command-line options `--domain-id`, `--namespace`, and `--cyclonedds-xml`
override the corresponding robot transport settings in `params.yaml`; the ROS
source domain is set separately in `command.yaml`.

## Policy directories

[`config/gmt_v1.0.0`](config/gmt_v1.0.0) is the bundled GMT motion-tracking policy.
Copy it to create a policy with its own model and configuration:

```text
my_policy/
├── params.yaml       # transport, control rate, startup, and policy paths
├── policy.yaml       # policy rate, manager files, and ONNX tensor contracts
├── robot.yaml        # all 31 SDK joints, IMU transform, named frames
├── command.yaml      # registered command terms and their sources
├── observation.yaml  # ordered ONNX input groups
├── action.yaml       # named action terms, ordering, gains, and limits
├── termination.yaml  # policy termination terms
├── event.yaml        # startup and reset event terms
├── policy.onnx       # policy model
└── motion.onnx       # optional recorded motion source
```

Paths inside a YAML file are resolved relative to that YAML file. Names under
`policy.inputs` must equal the ONNX input names. Each `policy.outputs` entry
binds a named action output to a slice of an ONNX tensor.

Action configuration follows the controller manager layout:

```yaml
igris_c_inference_actions:
  apply_order: [joint_position]
  terms:
    joint_position:
      function: safe_joint_position
      output: actions
      params:
        unselected_joint_behavior: hold_current
      action_joints: [LSP, RSP]
      action_scale: [0.25, 0.25]
      kp: [100.0, 100.0]
      kd: [5.0, 5.0]
      output_clip: {min: -100.0, max: 100.0}
      lpf_cutoff_hz: 12.8
```

The registry provides `joint_position`, `safe_joint_position`, and
`motion_command_action`. The bundled profile uses `safe_joint_position` for
23 policy joints and `motion_command_action` for six wrist and two neck joints.
`safe_joint_position` can set `params.unselected_joint_behavior` to
`default` or `hold_current`. `motion_command_action` consumes no policy output.
It copies a configured slice of a command signal into selected joint position
targets and applies the `kp` and `kd` values configured for that action term.

### Command terms

Each entry under `terms` has a registered `function`, one `source`, and
term-specific `params`. Observations retrieve its processed vector by name:

```yaml
igris_c_inference_commands:
  terms:
    velocity:
      function: velocity_command
      source:
        type: dds
        params: {topic: rt/cmd_vel}
      params:
        stale_after_ms: 200
        on_stale: zero
        components:
          - {field: linear_x, scale: 1.0, min: -0.6, max: 0.6}
          - {field: linear_y, scale: 1.0, min: -0.5, max: 0.5}
          - {field: angular_z, scale: 1.0, min: -1.0, max: 1.0}
```

```yaml
- function: command
  params: {name: velocity}
  update_at: policy
```

The generic `motion_reference` composes configured joint, body,
anchor-velocity, orientation, and height fields. Without command history
parameters it produces one current frame, and observation terms can select
independent ranges and own their histories:

```yaml
- function: command
  params: {name: motion, offset: 23, size: 39}
  history_length: 10
  update_at: policy
```

Keeping history in the observation term allows each motion field to have its
own history contract. Each stored orientation remains relative to the robot
pose used when that observation was created.

Set `params.history_length`, `params.history_stride`, and
`params.history_order` on `motion_reference` when the command must own the
source-frame window. The defaults are `1`, `1`, and `oldest_to_newest`.
Command-owned history is packed component-major and evaluates every selected
frame with the current robot state. Consequently, an orientation history is
expressed entirely relative to the current robot frame, matching
`sonic_motion_reference`. Do not also configure observation history for an
already-stacked command component.

Both motion-reference commands accept `params.heading_alignment`. The default
`raw` mode preserves source world orientations. `initial` latches the robot-to-
reference yaw on the first successfully prepared current frame and applies
that fixed world-yaw rotation to subsequent reference orientations. It removes
an arbitrary world-heading offset without removing the motion's relative yaw,
roll, or pitch trajectory. Resetting the command clears the latch. The same
behavior applies to ONNX replay and ROS motion sources.

An unsupported source type is a configuration error and stops startup. See
[`docs/WALK_TWIST.md`](docs/WALK_TWIST.md) for the DDS wire contract.
Policies without commands may use `terms: {}` and omit `command` observations.

### Observation terms and history

Terms are concatenated in YAML order. A term may define:

- `params`: inputs consumed by its registered factory;
- `history_length`: number of selected policy-rate samples, default `1`;
- `history_stride`: distance between selected samples, default `1`;
- `history_order`: `oldest_to_newest` (default) or `newest_to_oldest`;
- `lpf_cutoff_hz`: first-order filter cutoff at `control_hz`, default `-1`.
- `update_at`: `control` (default) or `policy`; command terms whose values are
  prepared by a policy step should use `policy`.

For example, `history_length: 3`, `history_stride: 2`, and the default order
produces `[t-4, t-2, t]`. At reset, the first value fills the entire history,
matching Isaac Lab's observation-history behavior.

Built-in generic terms include `command`, `base_ang_vel`,
`projected_gravity`, `joint_pos_rel`, `joint_vel`, and `last_actions`. Named
robot-frame terms are `frame_ang_vel` and `frame_projected_gravity`.
`last_actions` requires `params.name` and reads the named term's raw ONNX
action before clipping, filtering, scaling, or safety processing.

## GMT v1.0.0 motion tracking

[`config/gmt_v1.0.0`](config/gmt_v1.0.0) contains the policy configuration and
the bundled `policy.onnx`. It accepts one `obs` input of 1968 values and
produces 23 `actions` values. The 50 Hz policy observation is assembled in this
order:

| Input segment | Size | Values |
|---|---:|---|
| Joint position history | 460 | 20 samples of 23 relative joint positions |
| Joint velocity history | 460 | 20 samples of 23 SDK joint velocities |
| Projected gravity history | 60 | 20 samples of 3 values |
| Base angular velocity history | 60 | 20 samples of 3 values |
| Previous action history | 460 | 20 samples of 23 raw policy actions |
| Filtered motion reference | 468 | Six frames of joint positions, anchor height and orientation, body positions, and anchor velocities |

The `filtered` command in [`command.yaml`](config/gmt_v1.0.0/command.yaml)
uses six ROS frames with stride two, ordered oldest to newest. At the producer's
100 Hz rate, the window spans ten frame intervals and requires 11 valid frames
before the command is ready. `heading_alignment: initial` removes the initial
world-yaw offset while preserving subsequent relative turns. Its stale-source
behavior is `hold_last`.

The included `motion.onnx` is not used by this configuration: both command
terms select `source.type: ros`. To replay recorded motion instead, change the
source of both terms to a compatible ONNX motion file and check its quaternion
order and metadata. The bundled `policy.onnx` and `policy.yaml` are a matched
pair; a different policy export also requires a matching observation and
action configuration.

Run the bundled policy with:

```bash
pixi run --locked run \
  --config "$PWD/config/gmt_v1.0.0/params.yaml"
```

## Wrist and neck reference actions

The bundled policy predicts 23 joint targets. The `motion_command` action
reads eight more joint positions from the live ROS reference and applies them
to both wrists and the neck. These joints are part of the 31-joint SDK model
but are not ONNX action outputs:

| Action term | Joints | Source |
|---|---:|---|
| `joint_position` | 23 | `actions` from `policy.onnx` |
| `motion_command` | 8 | `motion_command` ROS reference |

The action order and gains are in [`action.yaml`](config/gmt_v1.0.0/action.yaml).
The bundled profile uses `safe_joint_position` for the policy joints and
`motion_command_action` for wrist and neck targets. Before low-level control,
place the robot in a pose compatible with the reference. The node does not
perform a stand-to-motion interpolation.

### Live ROS motion reference

Both command terms in the bundled [`command.yaml`](config/gmt_v1.0.0/command.yaml)
subscribe to `/gmr/teleop/retarget_frame` on ROS domain `0` with best-effort
QoS. The teleop producer defaults to ROS domain `91`, so start it with
`--ros_domain_id 0` or change **both** `domain_id` entries in `command.yaml`
to `91`. The robot DDS domain is configured separately in `params.yaml`.

```bash
# From the igris_c_teleop_public repository, using the bundled robot namespace:
pixi run teleop -- --dds_namespace igris_c_f90b76f7 \
  --dds_domain_id 0 --ros_domain_id 0 --target robot
```

The ROS `std_msgs/msg/String` payload must contain `dof_pos`, `dof_vel`,
`root_rot` (wxyz), `link_body_list`, and `keybody_rot_local` (wxyz).
`keybody_pos_local`, `root_pos`, `root_vel`, and `root_angvel` supply the body
position, anchor height, and velocity components used by the bundled policy.
The `joint_names` lists in `command.yaml` describe the 31 incoming joint
positions and velocities, including both wrist triplets and neck yaw/pitch.
The `filtered` command maps 23 joints to the policy observation; the separate
`motion_command` term maps the other eight joints to the output action.

### VR Motion Viewer

The standalone, read-only viewer subscribes to the teleop ROS topic and draws
the moving robot keybody skeleton in 3D. Drag to orbit and use the wheel or
pinch to zoom. It applies `root_pos` and `root_rot` to `keybody_pos_local`,
then streams the resulting points to the browser. It uses Python's HTTP server
plus `rclpy` and `std_msgs`, which are already in the Pixi environment. It does
not start the inference node or publish commands.

```bash
ROS_DOMAIN_ID=0 pixi run --locked joint-viewer
```

Open `http://127.0.0.1:8765`. Match `ROS_DOMAIN_ID` or `--ros-domain-id` to the
teleop producer's `--ros_domain_id`; this release's ROS source uses domain 0.
Use `--topic` to change the
ROS topic, or `--host 0.0.0.0 --port 8765` to open the page from another
machine. The viewer is in `tools/vr_motion_viewer.py` and can also be run
directly with the Pixi Python interpreter. It shows the streamed keybody
positions as a skeleton, without a robot mesh.

## Extending the runtime

- Add observation terms under `modules/policy/mdp/observations` and register
  them in `ObservationRegistry.cpp`.
- Add action terms under `modules/policy/mdp/actions` and register them in
  `ActionRegistry.cpp`.
- Add command terms under `modules/policy/mdp/commands` and register them in
  `CommandRegistry.cpp`.
- Add lifecycle behavior through `EventManager` or `TerminationManager`, not
  application-specific branches in `main.cpp`.
- Put robot-specific serial frame definitions in `robot.yaml` rather than in
  generic observation code.

Configuration is validated at startup. Motion-reference profiles use
`on_stale: hold_last`, so a frozen source retains its latest history while
policy inference continues with current robot state. `reject` remains
available for deployments that require stale input to stop policy advancement.
See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for a compact field reference.
