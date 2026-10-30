# Configuration reference

Start from `config/example_policy` for velocity policies,
`config/fsq_gmt_lpf` for multi-motion FSQ tracking, or
`config/sonic_any2any_lpf` for the exported SONIC+LoRA tracker. Relative paths
are resolved from the YAML file that contains them.

## `params.yaml`

- `robot`: DDS domain, robot namespace, CycloneDDS XML, and LowState/LowCmd
  topics.
- `loop.control_hz`: state filtering, action processing, and LowCmd rate.
- `loop.state_timeout_ms`: maximum LowState age while `RUNNING`, default 100 ms.
- `startup.enter_low_level_mode`: whether startup requests low-level joint
  control once after the first state and command source are ready. It defaults
  to `true`. When `false`, the node remains `IDLE` until another component
  changes the robot to low-level mode. LowCmd is published in `RUNNING`.
- `startup.low_level_mode_timeout_ms`: deadline for the SDK mode request.
  `first_state_timeout_ms` and `command_timeout_ms` are accepted legacy keys;
  the initial `WAIT` and command-source wait have no deadline.
- `files.robot`: robot configuration path.
- `files.policy_config`: `policy.yaml` path.
- `files.policy`: ONNX model path.

## `policy.yaml`

- `rate_hz`: history sampling, command advancement, and ONNX rate.
- `files`: command, observation, action, termination, and event manager files.
- `inputs`: ordered ONNX input tensor names.
- `input_shapes`: optional explicit input shapes.
- `outputs`: named bindings with `tensor`, `offset`, and `size`.

Each observation group name must match an ONNX input name. Every action term's
`output` must match one named model output binding with the same size.

The FSQ GMT policies must expose ONNX inputs named
`actor_observations` (750 values) and `motion_reference` (520 values), and an
action output with 23 values. The legacy GMT `obs`/`encoder_obs` policy is not
compatible with this configuration.

## `robot.yaml`

`robot_joints` must uniquely cover SDK indices 0 through 30. A joint's `name`
is the concise deployment name; `model_joint` is its URDF/training name. The
file also defines the IMU mounting transform, gravity direction, default
commands, gains, joint limits, and the five kinematic modes.

Named `frames` describe a serial rotation chain from `base_link`:

```yaml
frames:
  pelvis:
    joints:
      - {joint: WY, axis: [0.0, 0.0, 1.0]}
      - {joint: WR, axis: [1.0, 0.0, 0.0]}
      - {joint: WP, axis: [0.0, 1.0, 0.0]}
```

Frame-aware observations use these definitions to derive angular velocity and
projected gravity without assuming the articulation root is the pelvis.

## `command.yaml`

`terms` is a name-to-command map. Every entry selects a registered command
function, a supported source, and term-specific parameters. The generic
`command` observation reads a processed signal by `name`.

The built-in terms are:

- `velocity_command`: DDS `WalkTwist`, with configurable component order,
  scaling, clipping, timeout, and stale behavior;
- `motion_reference`: configurable joint, body, anchor-velocity, orientation,
  and height fields from an ONNX or ROS source. It returns one current frame by
  default. Optional `history_length`, `history_stride`, and `history_order`
  parameters make the command select and pack a source-frame window;
- `sonic_motion_reference`: recorded ONNX or ROS motion frames, alignment,
  source-specific future/history selection, and SONIC's packed reference
  vector.

For an ONNX motion source, `params.quaternion_order` may be `wxyz` or `xyzw`;
it defaults to `wxyz` for backward compatibility. Set it to `xyzw` for motion
files exported by the current Isaac Lab motion exporter. ROS motion payload
quaternions use WXYZ and are not affected by this ONNX-only setting.

Motion-reference terms default to `params.heading_alignment: raw`. Set it to
`initial` to latch the first current robot-to-reference yaw and rotate later
reference orientations by that fixed offset. Command reset clears the latch;
joint, position, velocity, height, and action fields are unaffected.

History configured on an observation term stores values after each observation
has been computed, so stored orientations use the robot pose from their own
update times. History configured on `motion_reference` is component-major and
computes every selected source-frame orientation using the current robot pose,
matching `sonic_motion_reference`. Leave command history at its one-frame
default for observation-owned history, and do not apply observation history to
an already-stacked command component.

Source types are validated by the selected command. A velocity command does
not accept ONNX, for example, and exits at startup instead of silently using a
different interpretation.

`params.on_stale` controls stale source behavior. The bundled motion-reference
profiles use `hold_last`, which keeps inference running with the latest valid
command history and current robot state when a live source freezes. `reject`
aborts that policy step instead. Velocity commands also support `zero`.

## `observation.yaml`

`obs_joints` controls the order used by generic `joint_pos_rel` and
`joint_vel`. Terms within a group are concatenated in YAML order.

```yaml
- function: joint_vel
  history_length: 3
  history_stride: 2
  history_order: oldest_to_newest
  lpf_cutoff_hz: 16
```

This example selects `[t-4, t-2, t]`. `history_order` may be
`newest_to_oldest`; `history_length` and `history_stride` default to 1.
`lpf_cutoff_hz` defaults to -1 (disabled) and otherwise updates at
`control_hz`. Histories advance at `policy_hz`. `update_at` defaults to
`control`; set it to `policy` for a term generated during policy preparation,
such as the SONIC motion-reference command.

Available generic functions are:

- `command`, with `params.name` and optional `offset` plus `size` for a
  bounded vector slice;
- `base_ang_vel` and `projected_gravity`;
- `joint_pos_rel` and `joint_vel`;
- `last_actions`, with `params.name` selecting an action term;
- `frame_ang_vel` and `frame_projected_gravity`, with `params.frame_name`;
- `aligned_joint_pos_rel` and `aligned_joint_vel`, with `params.alignment`;
- `aligned_last_actions`, with `params.name` and `params.alignment`.

## `action.yaml`

Actions use the same manager structure and registry names as the controller:

```yaml
igris_c_inference_actions:
  apply_order: [joint_position]
  terms:
    joint_position:
      function: safe_joint_position
      output: actions
      params: {}
      action_joints: [LSP, RSP]
      action_scale: [0.25, 0.25]
      kp: [100.0, 100.0]
      kd: [5.0, 5.0]
      output_clip: {min: -100.0, max: 100.0}
      lpf_cutoff_hz: 12.8
```

`action_joints` defines each term's action order. `action_scale`, `kp`, and
`kd` have one entry per selected joint. Both `joint_position` and
`safe_joint_position` use:

```text
target = robot_default + processed_action * action_scale
```

`lpf_cutoff_hz` optionally applies an exponential first-order low-pass filter
after clipping. It defaults to `-1` (disabled) and otherwise must be positive:

```text
previous_weight = exp(-2 * pi * cutoff_hz / policy_hz)
filtered = previous_weight * previous + (1 - previous_weight) * clipped
```

The first action passes through unchanged. `last_actions` reads the unfiltered,
unclipped raw ONNX output from the term named by `params.name`, matching Isaac
Lab's `last_action` observation.

Unselected joints receive their defaults and gains from `robot.yaml`.
Set `params.unselected_joint_behavior: hold_current` to retain their latest
measured positions instead; this is used for SONIC's unmodeled wrists and neck.

`motion_command_action` consumes no ONNX output. It reads `params.command`,
selects one contiguous block beginning at `params.offset`, and writes those
values to the joints listed in `params.joints`. The `joints`, `kp`, and `kd`
vectors must have equal lengths. Put the term after the policy action in
`apply_order` when its selected joints must override the policy command.

`params.position_limit_recovery` optionally enables the original GMT behavior
that progressively pulls targets back to hard limits after the measured joint
passes a configured soft-limit ratio. `excluded_joints` identifies parallel
joints whose limits remain the controller's responsibility:

```yaml
position_limit_recovery:
  soft_limit_ratio: 0.8
  excluded_joints: [WR, WP, LAP, LAR, RAP, RAR]
```

## Startup and runtime behavior

Startup validates YAML, command/source compatibility, joint coverage, ONNX
inputs and output size. The process enters `WAIT` until it receives valid
LowState and RobotState, then enters `IDLE`. There is no first-state or
command-source deadline. Once commands are ready, the node checks one inference
and command before optionally requesting low-level mode. It enters `RUNNING`
only after RobotState confirms that mode. Leaving low-level mode returns to
`IDLE`; re-entry waits for a fresh state and new command data. A policy stop
also returns to `IDLE` and requires the robot to leave and re-enter low-level
mode before another run. A stale LowState enters `ERROR`; a fresh, valid
LowState returns the node to `IDLE` after a periodic check. The robot must
leave and re-enter low-level mode before another run. Other `ERROR` causes
remain latched until the process is restarted.
The inference node observes mode changes through RobotState.

A policy step prepares all commands, samples observation history, runs ONNX,
and commits command advancement only after successful inference. During
`RUNNING`, invalid or stale LowState, inference or command failure, and LowCmd
publication failure enter `ERROR` without replaying the previous LowCmd.

### Diagnostics

Startup errors identify the component and configuration or model path. Runtime
errors identify the failed stage and, where applicable, the command term,
observation group and term, tensor, output binding, vector index, and expected
versus actual size. Identical consecutive failures are printed once; a changed
failure is printed immediately. Recovery reports the number of failed steps or
control cycles and repeats the last error.

Manager and runtime calls return an error code and populate their required
`std::string &error` output. `RlEnvModule` reports changed failures once and
reports recovery after successful processing resumes.

## First-run checklist

- Disable IGRIS-C controller gravity compensation.
- Test the exact policy directory in simulation.
- Match DDS domain, namespace, topics, and ROS domain where used.
- Inspect ONNX input names/dimensions and action output size.
- Confirm observation, action, and motion-source joint orders by name.
- Confirm default poses, action scales, gains, frame axes, and policy rates.
- Mechanically support the robot and keep the emergency stop available.
