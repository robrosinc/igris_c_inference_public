# DDS WalkTwist command contract

This document is the complete command publisher contract for
`igris_c_inference_public`. A publisher does not need access to another IGRIS-C
source tree. Generate bindings from the installed
`interfaces/WalkTwist.idl`, or reproduce the same DDS type in the publisher's
language.

## DDS identity

- Type: `igris_c::msg::dds::WalkTwist`
- Configured topic: normally `rt/cmd_vel`
- Wire topic with namespace: `<robot namespace>/rt/cmd_vel`
- QoS: reliable, volatile, keep-last, depth 10
- DDS domain: the `domain_id` in `params.yaml`

The publisher and inference process must use the same domain, wire topic, type
name, and compatible QoS.

## Fields and units

`WalkTwist` contains a `Header` and a `Twist`. `Twist` contains two
three-dimensional vectors:

| Field | Unit | Typical meaning |
| --- | --- | --- |
| `twist.linear.x` | m/s | forward velocity |
| `twist.linear.y` | m/s | lateral velocity |
| `twist.linear.z` | m/s | vertical velocity |
| `twist.angular.x` | rad/s | roll rate |
| `twist.angular.y` | rad/s | pitch rate |
| `twist.angular.z` | rad/s | yaw rate |

Only fields listed in a velocity term's
`command.yaml` `params.components` are consumed. Their YAML order is the
processed command-vector order. The example policy consumes forward, lateral,
and yaw commands.

The header is available for publisher bookkeeping, but inference freshness is
measured from local DDS receive time. It does not depend on `header.sec` or
`header.nanosec`. Publishers should still increment `header.seq` and may set a
meaningful timestamp and frame ID.

Publish continuously while control is active. A nominal 20 Hz rate is suitable
for the example's 200 ms stale timeout; choose a rate and timeout together for
the application.

## Minimal publisher procedure

1. Generate DDS language bindings from `WalkTwist.idl` without changing its
   module hierarchy or field types.
2. Create a participant in the configured DDS domain.
3. Create a reliable/volatile keep-last writer with depth 10.
4. Create the fully resolved topic, including the robot namespace.
5. Fill the selected velocity fields in SI units and publish continuously.

The IDL is the wire-level source of truth. `command.yaml` is the policy-level
source of truth for selected fields, scaling, clipping, order, and stale
behavior.
