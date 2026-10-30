#!/usr/bin/env python3
"""Replay an Isaac policy snapshot target through igris_c_inference_public.

This diagnostic is intentionally simulation-only. It creates a temporary ONNX
model whose action output reconstructs the snapshot's final joint-position
target, then launches the normal inference node. The normal ActionManager,
SafeJointPositionAction, LowCmd publisher, controller, and MuJoCo remain in the
runtime path.

The script does not request LOW_LEVEL_MODE. Start the simulation and controller,
and select LOW_LEVEL_MODE using the normal simulation procedure.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
import yaml

try:
    import onnx
    from onnx import TensorProto, checker, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
except ImportError as error:
    raise SystemExit(
        "This script requires the Python 'onnx' package. Run it from the "
        "newton_env environment used by robros_lab."
    ) from error


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = REPOSITORY_ROOT.parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config/fsq_gmt_lpf_pico_fullref_0915/params.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run igris_c_inference_public with a constant action that reproduces "
            "an Isaac policy snapshot's joint_position_target."
        )
    )
    parser.add_argument("--snapshot", required=True, type=Path, help="Isaac policy_io_*.npz snapshot")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Base inference params.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--unselected-joint-behavior",
        choices=("configured", "default", "hold_current"),
        default="configured",
        help=(
            "Behavior for the eight joints absent from the 23-DoF snapshot. "
            "Use 'default' for Isaac dummy-hand parity; 'configured' preserves action.yaml."
        ),
    )
    parser.add_argument("--domain-id", type=int, help="Override the DDS domain from params.yaml")
    parser.add_argument("--namespace", help="Override the DDS namespace from params.yaml")
    parser.add_argument("--cyclonedds-xml", type=Path, help="Override the CycloneDDS XML path")
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a simulation RobotState (default: 5)",
    )
    parser.add_argument(
        "--existing-lowcmd-window",
        type=float,
        default=1.0,
        help="Seconds to check for an existing DDS LowCmd publisher (default: 1)",
    )
    parser.add_argument(
        "--comparison-period",
        type=float,
        default=0.1,
        help="Seconds between LowCmd CSV samples (default: 0.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory for generated config, model, manifest, and LowCmd CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate artifacts without DDS checks or launching inference",
    )
    arguments = parser.parse_args()
    if arguments.domain_id is not None and arguments.domain_id < 0:
        parser.error("--domain-id must be non-negative")
    if arguments.state_timeout <= 0.0:
        parser.error("--state-timeout must be positive")
    if arguments.existing_lowcmd_window < 0.0:
        parser.error("--existing-lowcmd-window must be non-negative")
    if arguments.comparison_period <= 0.0:
        parser.error("--comparison-period must be positive")
    return arguments


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as error:
        raise RuntimeError(f"failed to read YAML '{path}': {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def resolve_path(base_file: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = base_file.parent / path
    return path.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_snapshot(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"metadata_json", "joint_position_target", "action_joint_names"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise RuntimeError(f"snapshot is missing arrays: {missing}")
            metadata = json.loads(str(archive["metadata_json"]))
            target = np.asarray(archive["joint_position_target"], dtype=np.float64).reshape(-1)
            joint_names = [str(name) for name in archive["action_joint_names"].reshape(-1)]
            done_after_step = bool(np.asarray(archive.get("done_after_step", False)).item())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"failed to load snapshot '{path}': {error}") from error

    if metadata.get("format") != "robros_policy_io_snapshot":
        raise RuntimeError(f"unsupported snapshot format: {metadata.get('format')!r}")
    if target.size != len(joint_names):
        raise RuntimeError(
            f"snapshot target has {target.size} values but {len(joint_names)} joint names"
        )
    if not np.isfinite(target).all():
        raise RuntimeError("snapshot joint_position_target contains a non-finite value")
    if len(set(joint_names)) != len(joint_names):
        raise RuntimeError("snapshot action_joint_names contains duplicates")
    return {
        "metadata": metadata,
        "target": target,
        "joint_names": joint_names,
        "done_after_step": done_after_step,
    }


def locate_action_term(action_root: dict[str, Any], output_name: str) -> tuple[str, dict[str, Any]]:
    config = action_root.get("igris_c_inference_actions")
    if not isinstance(config, dict):
        raise RuntimeError("action YAML lacks igris_c_inference_actions")
    terms = config.get("terms")
    if not isinstance(terms, dict):
        raise RuntimeError("action YAML lacks a terms mapping")
    matches = [
        (str(name), term)
        for name, term in terms.items()
        if isinstance(term, dict) and term.get("output") == output_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one action term consuming output '{output_name}', got {len(matches)}"
        )
    return matches[0]


def build_fixed_action(
    snapshot: dict[str, Any], action_term: dict[str, Any], robot_root: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    robot = robot_root.get("igris_c_inference_robot")
    if not isinstance(robot, dict) or not isinstance(robot.get("robot_joints"), list):
        raise RuntimeError("robot YAML lacks igris_c_inference_robot.robot_joints")
    joints = robot["robot_joints"]
    by_name: dict[str, dict[str, Any]] = {}
    for joint in joints:
        if not isinstance(joint, dict):
            raise RuntimeError("robot_joints entries must be mappings")
        name = str(joint.get("name", ""))
        model_joint = str(joint.get("model_joint", ""))
        if not name or not model_joint:
            raise RuntimeError("each robot joint needs name and model_joint")
        by_name[name] = joint

    action_joints = [str(name) for name in action_term.get("action_joints", [])]
    scales = np.asarray(action_term.get("action_scale", []), dtype=np.float64).reshape(-1)
    if not action_joints or scales.size != len(action_joints):
        raise RuntimeError("action_joints and action_scale must have the same non-zero size")
    if not np.isfinite(scales).all() or np.any(scales == 0.0):
        raise RuntimeError("action_scale must contain finite, non-zero values")

    snapshot_indices = {name: index for index, name in enumerate(snapshot["joint_names"])}
    targets: list[float] = []
    defaults: list[float] = []
    rows: list[dict[str, Any]] = []
    for action_index, action_joint in enumerate(action_joints):
        if action_joint not in by_name:
            raise RuntimeError(f"action joint '{action_joint}' is absent from robot.yaml")
        joint = by_name[action_joint]
        model_joint = str(joint["model_joint"])
        snapshot_index = snapshot_indices.get(model_joint)
        if snapshot_index is None:
            snapshot_index = snapshot_indices.get(action_joint)
        if snapshot_index is None:
            raise RuntimeError(
                f"snapshot lacks action joint '{action_joint}' (model joint '{model_joint}')"
            )
        target = float(snapshot["target"][snapshot_index])
        default = float(joint["default_position"])
        targets.append(target)
        defaults.append(default)
        rows.append(
            {
                "action_index": action_index,
                "action_joint": action_joint,
                "model_joint": model_joint,
                "robot_index": int(joint["robot_index"]),
                "snapshot_index": snapshot_index,
                "default_position": default,
                "action_scale": float(scales[action_index]),
                "target_position": target,
            }
        )

    target_array = np.asarray(targets, dtype=np.float64)
    default_array = np.asarray(defaults, dtype=np.float64)
    fixed_action = (target_array - default_array) / scales
    reconstructed = default_array + fixed_action * scales
    error = float(np.max(np.abs(reconstructed - target_array)))
    if error > 1.0e-12:
        raise RuntimeError(f"failed to reconstruct snapshot targets: max error {error}")

    clip = action_term.get("output_clip", {})
    if isinstance(clip, dict) and "min" in clip and "max" in clip:
        clip_min = float(clip["min"])
        clip_max = float(clip["max"])
        if np.any(fixed_action < clip_min) or np.any(fixed_action > clip_max):
            raise RuntimeError(
                "fixed action would be changed by action output_clip; exact target replay is impossible"
            )
    for row, action in zip(rows, fixed_action, strict=True):
        row["fixed_policy_action"] = float(action)
    return fixed_action.astype(np.float32), target_array, rows


def build_constant_policy(
    source_model_path: Path,
    output_path: Path,
    output_tensor_name: str,
    fixed_action: np.ndarray,
) -> None:
    source = onnx.load(source_model_path)
    source_inputs = [copy.deepcopy(value) for value in source.graph.input]
    source_input_names = [value.name for value in source_inputs]
    if not source_inputs:
        raise RuntimeError("source policy has no graph inputs")
    if output_tensor_name not in {value.name for value in source.graph.output}:
        raise RuntimeError(
            f"configured output tensor '{output_tensor_name}' is absent from source policy"
        )

    action_matrix = fixed_action.reshape(1, -1)
    initializer = numpy_helper.from_array(action_matrix, name="fixed_snapshot_action")
    output_info = helper.make_tensor_value_info(
        output_tensor_name, TensorProto.FLOAT, list(action_matrix.shape)
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", [initializer.name], [output_tensor_name])],
        "igris_c_snapshot_target_replay",
        source_inputs,
        [output_info],
        initializer=[initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="igris_c_snapshot_target_replay",
        opset_imports=[copy.deepcopy(item) for item in source.opset_import],
    )
    model.ir_version = source.ir_version
    checker.check_model(model)
    onnx.save(model, output_path)

    feeds: dict[str, np.ndarray] = {}
    for value in source_inputs:
        dimensions: list[int] = []
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(f"policy input '{value.name}' is not float32")
        for dimension in tensor_type.shape.dim:
            dimensions.append(int(dimension.dim_value) if dimension.dim_value > 0 else 1)
        feeds[value.name] = np.zeros(dimensions, dtype=np.float32)
    outputs = ReferenceEvaluator(model).run([output_tensor_name], feeds)
    actual = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    if source_input_names != [value.name for value in model.graph.input]:
        raise RuntimeError("generated policy changed the ONNX input contract")
    if not np.array_equal(actual, fixed_action):
        raise RuntimeError("generated policy did not return the exact fixed action")


def prepare_artifacts(arguments: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    snapshot_path = arguments.snapshot.expanduser().resolve()
    params_path = arguments.config.expanduser().resolve()
    if not snapshot_path.is_file():
        raise RuntimeError(f"snapshot does not exist: {snapshot_path}")
    if not params_path.is_file():
        raise RuntimeError(f"base inference config does not exist: {params_path}")

    params_root = read_yaml(params_path)
    inference = params_root.get("igris_c_inference")
    if not isinstance(inference, dict):
        raise RuntimeError("params YAML lacks igris_c_inference")
    files = inference.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("params YAML lacks igris_c_inference.files")

    robot_path = resolve_path(params_path, str(files["robot"]))
    policy_config_path = resolve_path(params_path, str(files["policy_config"]))
    source_policy_path = resolve_path(params_path, str(files["policy"]))
    for label, path in (
        ("robot", robot_path),
        ("policy config", policy_config_path),
        ("source policy", source_policy_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} file does not exist: {path}")

    policy_root = read_yaml(policy_config_path)
    policy = policy_root.get("igris_c_inference_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("policy YAML lacks igris_c_inference_policy")
    outputs = policy.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise RuntimeError("this diagnostic requires exactly one configured policy output")
    binding = outputs[0]
    if int(binding.get("offset", -1)) != 0:
        raise RuntimeError("this diagnostic requires policy output offset 0")
    output_name = str(binding.get("name", ""))
    output_tensor_name = str(binding.get("tensor", ""))
    output_size = int(binding.get("size", -1))
    if not output_name or not output_tensor_name or output_size <= 0:
        raise RuntimeError("policy output binding is incomplete")

    policy_files = policy.get("files")
    if not isinstance(policy_files, dict):
        raise RuntimeError("policy YAML lacks igris_c_inference_policy.files")
    manager_paths = {
        name: resolve_path(policy_config_path, str(configured))
        for name, configured in policy_files.items()
    }
    action_path = manager_paths.get("action")
    if action_path is None:
        raise RuntimeError("policy YAML lacks an action manager file")
    action_root = read_yaml(action_path)
    action_term_name, action_term = locate_action_term(action_root, output_name)
    robot_root = read_yaml(robot_path)
    snapshot = load_snapshot(snapshot_path)
    fixed_action, expected_target, rows = build_fixed_action(snapshot, action_term, robot_root)
    if fixed_action.size != output_size:
        raise RuntimeError(
            f"fixed action has {fixed_action.size} values but policy output size is {output_size}"
        )

    if arguments.unselected_joint_behavior != "configured":
        params = action_term.setdefault("params", {})
        if not isinstance(params, dict):
            raise RuntimeError(f"action term '{action_term_name}' params must be a mapping")
        params["unselected_joint_behavior"] = arguments.unselected_joint_behavior
    configured_unselected = action_term.get("params", {}).get(
        "unselected_joint_behavior", "default"
    )

    generated_action_path = output_dir / "action.yaml"
    generated_policy_config_path = output_dir / "policy.yaml"
    generated_model_path = output_dir / "fixed_target_policy.onnx"
    generated_params_path = output_dir / "params.yaml"
    manifest_path = output_dir / "manifest.json"
    expected_csv_path = output_dir / "expected_targets.csv"

    write_yaml(generated_action_path, action_root)
    generated_policy_root = copy.deepcopy(policy_root)
    generated_policy = generated_policy_root["igris_c_inference_policy"]
    generated_policy["files"] = {
        name: str(generated_action_path if name == "action" else path)
        for name, path in manager_paths.items()
    }
    write_yaml(generated_policy_config_path, generated_policy_root)
    build_constant_policy(
        source_policy_path, generated_model_path, output_tensor_name, fixed_action
    )

    generated_params_root = copy.deepcopy(params_root)
    generated_inference = generated_params_root["igris_c_inference"]
    generated_inference["files"] = {
        "robot": str(robot_path),
        "policy_config": str(generated_policy_config_path),
        "policy": str(generated_model_path),
    }
    startup = generated_inference.setdefault("startup", {})
    if not isinstance(startup, dict):
        raise RuntimeError("igris_c_inference.startup must be a mapping")
    startup["enter_low_level_mode"] = False
    write_yaml(generated_params_path, generated_params_root)

    with expected_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    network = generated_inference.get("robot")
    if not isinstance(network, dict):
        raise RuntimeError("igris_c_inference.robot must be a mapping")
    domain_id = int(network.get("domain_id", 0))
    robot_namespace = str(network.get("namespace", ""))
    cyclonedds_xml = str(network.get("cyclonedds_xml_path", ""))
    if arguments.domain_id is not None:
        domain_id = arguments.domain_id
    if arguments.namespace is not None:
        robot_namespace = arguments.namespace
    if arguments.cyclonedds_xml is not None:
        cyclonedds_xml = str(arguments.cyclonedds_xml.expanduser().resolve())

    metadata = snapshot["metadata"]
    manifest = {
        "mode": "simulation_only_fixed_snapshot_target",
        "snapshot": str(snapshot_path),
        "snapshot_sha256": sha256(snapshot_path),
        "snapshot_time_seconds": metadata.get("actual_time_seconds"),
        "snapshot_done_after_step": snapshot["done_after_step"],
        "base_params": str(params_path),
        "source_policy": str(source_policy_path),
        "source_policy_sha256": sha256(source_policy_path),
        "generated_policy": str(generated_model_path),
        "generated_policy_sha256": sha256(generated_model_path),
        "action_term": action_term_name,
        "output_name": output_name,
        "output_tensor": output_tensor_name,
        "unselected_joint_behavior": configured_unselected,
        "domain_id": domain_id,
        "namespace": robot_namespace,
        "cyclonedds_xml_path": cyclonedds_xml,
        "expected_targets": rows,
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    return {
        "params_path": generated_params_path,
        "manifest": manifest,
        "rows": rows,
        "expected_target": expected_target,
        "domain_id": domain_id,
        "namespace": robot_namespace,
        "cyclonedds_xml_path": cyclonedds_xml,
        "robot_state_topic": str(network.get("robot_state_topic", "rt/robotstate")),
        "command_topic": str(network.get("command_topic", "rt/lowcmd")),
    }


def import_sdk() -> Any:
    build_root = WORKSPACE_ROOT / "build"
    if build_root.is_dir():
        sys.path.insert(0, str(build_root))
    try:
        import igris_c_sdk as sdk
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "failed to import the CycloneDDS Python SDK. Source the workspace "
            f"setup and ensure {build_root / 'igris_c_sdk'} contains igris_c_sdk_py: {error}"
        ) from error
    return sdk


def load_cyclonedds_xml(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text.removeprefix("file://")).expanduser().resolve()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"failed to read CycloneDDS XML '{path}': {error}") from error


class DdsGuardAndMonitor:
    def __init__(
        self,
        sdk: Any,
        prepared: dict[str, Any],
        csv_path: Path,
        comparison_period: float,
    ) -> None:
        self.sdk = sdk
        self.prepared = prepared
        self.rows = prepared["rows"]
        self.expected = np.asarray(prepared["expected_target"], dtype=np.float64)
        self.comparison_period = comparison_period
        self.robot_state_event = threading.Event()
        self.lowcmd_event = threading.Event()
        self.monitor_enabled = False
        self.environment: Any | None = None
        self.control_state: Any | None = None
        self.start_time = time.monotonic()
        self.last_sample_time = 0.0
        self.last_report_time = 0.0
        self.sample_count = 0
        self.lock = threading.Lock()
        self.csv_stream = csv_path.open("w", encoding="utf-8", newline="")
        fieldnames = ["elapsed_seconds", "sequence", "max_abs_error"]
        for row in self.rows:
            name = row["action_joint"]
            fieldnames.extend((f"{name}_expected", f"{name}_lowcmd", f"{name}_error"))
        self.csv_writer = csv.DictWriter(self.csv_stream, fieldnames=fieldnames)
        self.csv_writer.writeheader()

        factory = sdk.ChannelFactory.Instance()
        factory.Init(
            prepared["domain_id"],
            prepared["namespace"],
            load_cyclonedds_xml(prepared["cyclonedds_xml_path"]),
        )
        if not factory.IsInitialized():
            raise RuntimeError("failed to initialize the DDS ChannelFactory")
        self.factory = factory
        self.robot_state_subscriber = sdk.RobotStateSubscriber(prepared["robot_state_topic"])
        self.lowcmd_subscriber = sdk.LowCmdSubscriber(
            prepared["command_topic"], sdk.QosProfile.SensorData()
        )
        if not self.robot_state_subscriber.init(self._robot_state_callback):
            raise RuntimeError("failed to initialize RobotState subscriber")
        if not self.lowcmd_subscriber.init(self._lowcmd_callback):
            raise RuntimeError("failed to initialize LowCmd subscriber")

    def _robot_state_callback(self, message: Any) -> None:
        self.environment = message.environment()
        self.control_state = message.state()
        self.robot_state_event.set()

    def _lowcmd_callback(self, message: Any) -> None:
        self.lowcmd_event.set()
        if not self.monitor_enabled:
            return
        now = time.monotonic()
        with self.lock:
            if now - self.last_sample_time < self.comparison_period:
                return
            self.last_sample_time = now
        motors = message.motors()
        actual = np.asarray(
            [float(motors[row["robot_index"]].q()) for row in self.rows], dtype=np.float64
        )
        errors = actual - self.expected
        maximum = float(np.max(np.abs(errors)))
        record: dict[str, Any] = {
            "elapsed_seconds": now - self.start_time,
            "sequence": int(message.header().seq()),
            "max_abs_error": maximum,
        }
        for row, expected, value, error in zip(
            self.rows, self.expected, actual, errors, strict=True
        ):
            name = row["action_joint"]
            record[f"{name}_expected"] = expected
            record[f"{name}_lowcmd"] = value
            record[f"{name}_error"] = error
        with self.lock:
            self.csv_writer.writerow(record)
            self.csv_stream.flush()
            self.sample_count += 1
            should_report = now - self.last_report_time >= 1.0
            if should_report:
                self.last_report_time = now
        if should_report:
            worst = int(np.argmax(np.abs(errors)))
            print(
                "[LowCmd] selected-joint max_abs_error="
                f"{maximum:.9g} rad at {self.rows[worst]['action_joint']}",
                flush=True,
            )

    def verify_simulation(self, timeout: float) -> None:
        if not self.robot_state_event.wait(timeout):
            raise RuntimeError(
                f"timed out waiting for RobotState after {timeout:g} s; inference will not start"
            )
        if self.environment != self.sdk.ROBOT_ENV_SIM:
            raise RuntimeError(
                f"RobotState environment is {self.environment}, not ROBOT_ENV_SIM; refusing to publish"
            )
        print(f"Verified simulation RobotState; control state is {self.control_state}.")

    def verify_no_existing_lowcmd(self, duration: float) -> None:
        self.lowcmd_event.clear()
        if duration > 0.0 and self.lowcmd_event.wait(duration):
            raise RuntimeError(
                "an existing DDS LowCmd publisher is active; stop it before running this diagnostic"
            )

    def enable_monitor(self) -> None:
        self.lowcmd_event.clear()
        self.start_time = time.monotonic()
        self.monitor_enabled = True

    def close(self) -> None:
        self.monitor_enabled = False
        for subscriber_name in ("lowcmd_subscriber", "robot_state_subscriber"):
            subscriber = getattr(self, subscriber_name, None)
            if subscriber is not None:
                subscriber.stop()
        self.csv_stream.close()
        factory = getattr(self, "factory", None)
        if factory is not None:
            factory.Release()


def launch_inference(arguments: argparse.Namespace, prepared: dict[str, Any]) -> int:
    command = [
        "ros2",
        "run",
        "igris_c_inference_public",
        "igris_c_inference_node",
        "--config",
        str(prepared["params_path"]),
    ]
    if arguments.domain_id is not None:
        command.extend(("--domain-id", str(arguments.domain_id)))
    if arguments.namespace is not None:
        command.extend(("--namespace", arguments.namespace))
    if arguments.cyclonedds_xml is not None:
        command.extend(("--cyclonedds-xml", str(arguments.cyclonedds_xml.expanduser().resolve())))

    print("Launching:", " ".join(command), flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        print("Stopping snapshot target replay...", flush=True)
        os.killpg(process.pid, signal.SIGINT)
        try:
            return process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            return process.wait(timeout=5.0)


def create_output_directory(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="igris_c_snapshot_target_"))
    output = requested.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"--output-dir must not already exist: {output}") from error
    return output


def main() -> int:
    arguments = parse_arguments()
    try:
        output_dir = create_output_directory(arguments.output_dir)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Diagnostic artifacts: {output_dir}")
    monitor: DdsGuardAndMonitor | None = None
    try:
        prepared = prepare_artifacts(arguments, output_dir)
        manifest = prepared["manifest"]
        print(
            f"Prepared {len(prepared['rows'])} selected-joint targets from snapshot "
            f"time {manifest['snapshot_time_seconds']} s."
        )
        print(
            "Unselected joints: "
            f"{manifest['unselected_joint_behavior']}; LOW_LEVEL_MODE will not be requested."
        )
        print(f"Expected target table: {output_dir / 'expected_targets.csv'}")
        if arguments.dry_run:
            print("Dry run completed; inference was not launched.")
            return 0

        sdk = import_sdk()
        monitor = DdsGuardAndMonitor(
            sdk,
            prepared,
            output_dir / "lowcmd_comparison.csv",
            arguments.comparison_period,
        )
        monitor.verify_simulation(arguments.state_timeout)
        monitor.verify_no_existing_lowcmd(arguments.existing_lowcmd_window)
        monitor.enable_monitor()
        result = launch_inference(arguments, prepared)
        if result != 0:
            print(f"igris_c_inference_node exited with status {result}", file=sys.stderr)
        return result
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if monitor is not None:
            monitor.close()
        print(f"Artifacts retained in: {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
