#!/usr/bin/env python3
"""Run one policy-free Isaac Sim 6.1.0 episode and record schema v0.1.

The module deliberately keeps Isaac Sim imports inside ``run_episode``.  This
lets a normal Python interpreter validate a local experiment config before the
Isaac runtime is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


EXPERIMENT_SCHEMA = "fragilegrab.experiment/0.1"
EPISODE_SCHEMA = "fragilegrab.episode/0.1"
ISAAC_VERSION = "6.1.0"
EPISODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
URI_PREFIXES = ("omniverse://", "http://", "https://")


class ConfigError(ValueError):
    """The experiment config is incomplete or incompatible with this runner."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise ConfigError("PyYAML is required to read the experiment config") from exc

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError("experiment config root must be a mapping")
    return value


def _at(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"missing required field: {dotted_path}")
        value = value[part]
    return value


def _required(config: dict[str, Any], dotted_path: str) -> Any:
    value = _at(config, dotted_path)
    if value is None or (isinstance(value, str) and value.strip().upper() == "TBD"):
        raise ConfigError(f"runtime field is not resolved: {dotted_path}")
    return value


def _positive_number(value: Any, dotted_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{dotted_path} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ConfigError(f"{dotted_path} must be a positive finite number")
    return result


def _nonnegative_int(value: Any, dotted_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{dotted_path} must be a non-negative integer")
    return value


def _positive_int(value: Any, dotted_path: str) -> int:
    result = _nonnegative_int(value, dotted_path)
    if result == 0:
        raise ConfigError(f"{dotted_path} must be greater than zero")
    return result


def _prim_path(value: Any, dotted_path: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value == "/":
        raise ConfigError(f"{dotted_path} must be an absolute USD prim path")
    return value


def _runtime_path(value: str, config_path: Path) -> str:
    if value.startswith(URI_PREFIXES):
        return value
    candidate = Path(os.path.expandvars(os.path.expanduser(value)))
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return str(candidate.resolve())


def validate_config(
    config: dict[str, Any], config_path: Path, *, check_scene_exists: bool = True
) -> dict[str, Any]:
    """Validate the concrete fields needed by the policy-free runner."""

    if _required(config, "schema_version") != EXPERIMENT_SCHEMA:
        raise ConfigError(f"schema_version must be {EXPERIMENT_SCHEMA}")
    if _required(config, "source.kind") != "simulation":
        raise ConfigError("source.kind must be simulation")
    experiment_id = _required(config, "experiment.id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ConfigError("experiment.id must be a non-empty string")
    if _required(config, "source.simulator.name") != "isaac_sim":
        raise ConfigError("source.simulator.name must be isaac_sim")
    if str(_required(config, "source.simulator.version")) != ISAAC_VERSION:
        raise ConfigError(f"this runner only targets Isaac Sim {ISAAC_VERSION}")

    backend = _required(config, "source.simulator.physics_backend")
    if backend not in {"physx", "newton", "remotesim"}:
        raise ConfigError(
            "source.simulator.physics_backend must be physx, newton or remotesim"
        )
    device = _required(config, "source.simulator.device")
    renderer = _required(config, "source.simulator.renderer")
    if not isinstance(device, str) or not device.strip():
        raise ConfigError("source.simulator.device must be a non-empty string")
    if not isinstance(renderer, str) or not renderer.strip():
        raise ConfigError("source.simulator.renderer must be a non-empty string")
    physics_dt = _positive_number(
        _required(config, "source.simulator.physics_dt_s"),
        "source.simulator.physics_dt_s",
    )
    meters_per_unit = _positive_number(
        _required(config, "source.simulator.expected_stage_meters_per_unit"),
        "source.simulator.expected_stage_meters_per_unit",
    )
    seed = _required(config, "source.simulator.seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("source.simulator.seed must be an integer")

    scene_ref = _required(config, "source.simulator.scene_ref")
    if not isinstance(scene_ref, str) or not scene_ref.strip():
        raise ConfigError("source.simulator.scene_ref must be a non-empty string")
    resolved_scene = _runtime_path(scene_ref, config_path)
    if check_scene_exists and not resolved_scene.startswith(URI_PREFIXES):
        if not Path(resolved_scene).is_file():
            raise ConfigError(f"scene_ref does not exist: {resolved_scene}")
    asset_ref = _required(config, "source.simulator.robot_asset_ref")
    if not isinstance(asset_ref, str) or not asset_ref.strip():
        raise ConfigError("source.simulator.robot_asset_ref must be a stable reference")

    if _required(config, "robot.name") != "piper_x":
        raise ConfigError("robot.name must be piper_x")
    robot_prim = _prim_path(
        _required(config, "robot.articulation_prim_path"),
        "robot.articulation_prim_path",
    )
    dof_names = _required(config, "robot.expected_dof_names")
    if (
        not isinstance(dof_names, list)
        or not dof_names
        or any(not isinstance(name, str) or not name or name.upper() == "TBD" for name in dof_names)
        or len(set(dof_names)) != len(dof_names)
    ):
        raise ConfigError("robot.expected_dof_names must be a non-empty unique string list")
    joint_mapping_ref = _required(config, "robot.joint_mapping_ref")
    if not isinstance(joint_mapping_ref, str) or not joint_mapping_ref.strip():
        raise ConfigError("robot.joint_mapping_ref must be a non-empty string")

    for dotted_path in (
        "task.id",
        "task.instruction",
        "task.object_id",
        "task.initial_condition_ref",
    ):
        value = _required(config, dotted_path)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{dotted_path} must be a non-empty string")

    if _required(config, "policy.enabled") is not False:
        raise ConfigError("policy.enabled must remain false for this runner")
    if _required(config, "policy.name") != "pi05":
        raise ConfigError("policy.name must remain pi05")
    if _required(config, "execution.action_source") != "scripted_joint_position_targets":
        raise ConfigError("execution.action_source must be scripted_joint_position_targets")
    semantics = _required(config, "execution.action_semantics")
    expected_semantics = {
        "mode": "position",
        "value_type": "absolute",
        "rotation_unit": "rad",
        "translation_unit": "m",
    }
    if not isinstance(semantics, dict) or any(
        semantics.get(key) != value for key, value in expected_semantics.items()
    ):
        raise ConfigError(f"execution.action_semantics must equal {expected_semantics}")

    max_steps = _positive_int(
        _required(config, "recording.max_physics_steps"),
        "recording.max_physics_steps",
    )
    observation_period = _positive_int(
        _required(config, "recording.observation_every_physics_steps"),
        "recording.observation_every_physics_steps",
    )
    if max_steps % observation_period != 0:
        raise ConfigError(
            "recording.max_physics_steps must be divisible by "
            "recording.observation_every_physics_steps"
        )
    observation_rate_hz = 1.0 / (physics_dt * observation_period)
    warmup_physics_steps = _nonnegative_int(
        _required(config, "recording.warmup_physics_steps"),
        "recording.warmup_physics_steps",
    )
    warmup_render_frames = _nonnegative_int(
        _required(config, "recording.warmup_render_frames"),
        "recording.warmup_render_frames",
    )
    if _required(config, "recording.episode_schema_version") != EPISODE_SCHEMA:
        raise ConfigError(f"recording.episode_schema_version must be {EPISODE_SCHEMA}")
    if _required(config, "recording.preserve_per_source_timing") is not True:
        raise ConfigError("recording.preserve_per_source_timing must be true")
    required_streams = _required(config, "recording.required_streams")
    baseline_streams = {
        "observation.image.base_rgb",
        "observation.image.wrist_rgb",
        "robot_state",
        "object_state",
        "contact",
        "joint_effort",
    }
    if not isinstance(required_streams, list) or not baseline_streams.issubset(
        required_streams
    ):
        raise ConfigError(
            f"recording.required_streams must include {sorted(baseline_streams)}"
        )

    cameras: dict[str, dict[str, Any]] = {}
    for role in ("base_rgb", "wrist_rgb"):
        prefix = f"recording.cameras.{role}"
        prim_path = _prim_path(_required(config, f"{prefix}.prim_path"), f"{prefix}.prim_path")
        tick_rate = _positive_number(
            _required(config, f"{prefix}.tick_rate_hz"), f"{prefix}.tick_rate_hz"
        )
        if not math.isclose(tick_rate, observation_rate_hz, rel_tol=0.0, abs_tol=1e-9):
            raise ConfigError(
                f"{prefix}.tick_rate_hz must equal the derived observation rate "
                f"{observation_rate_hz} Hz"
            )
        resolution = _required(config, f"{prefix}.resolution_hw")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in resolution)
        ):
            raise ConfigError(f"{prefix}.resolution_hw must be [height, width]")
        cameras[role] = {
            "prim_path": prim_path,
            "tick_rate_hz": tick_rate,
            "resolution_hw": tuple(resolution),
        }

    object_prim = _prim_path(
        _required(config, "recording.object_state.prim_path"),
        "recording.object_state.prim_path",
    )
    contact_prim = _prim_path(
        _required(config, "recording.contact.sensor_prim_path"),
        "recording.contact.sensor_prim_path",
    )
    include_raw_contact = _required(config, "recording.contact.include_raw")
    if not isinstance(include_raw_contact, bool):
        raise ConfigError("recording.contact.include_raw must be boolean")

    video_enabled = _required(config, "recording.video.enabled")
    if video_enabled is not True:
        raise ConfigError("recording.video.enabled must be true for the baseline episode")
    video_fourcc = _required(config, "recording.video.fourcc")
    if not isinstance(video_fourcc, str) or len(video_fourcc) != 4:
        raise ConfigError("recording.video.fourcc must contain exactly four characters")

    commands = _required(config, "execution.scripted_commands")
    if not isinstance(commands, list) or not commands:
        raise ConfigError("execution.scripted_commands must contain at least one command")
    normalized_commands: list[dict[str, Any]] = []
    previous_step = -1
    for index, command in enumerate(commands):
        prefix = f"execution.scripted_commands[{index}]"
        if not isinstance(command, dict):
            raise ConfigError(f"{prefix} must be a mapping")
        step = command.get("at_physics_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step >= max_steps:
            raise ConfigError(f"{prefix}.at_physics_step must be in [0, max_physics_steps)")
        if step <= previous_step:
            raise ConfigError("scripted command steps must be strictly increasing")
        previous_step = step
        targets = command.get("joint_position_targets")
        if not isinstance(targets, dict) or not targets:
            raise ConfigError(f"{prefix}.joint_position_targets must be a non-empty mapping")
        unknown = set(targets) - set(dof_names)
        if unknown:
            raise ConfigError(f"{prefix} contains unknown DOFs: {sorted(unknown)}")
        normalized_targets: dict[str, float] = {}
        for name, target in targets.items():
            if isinstance(target, bool) or not isinstance(target, (int, float)):
                raise ConfigError(f"{prefix}.joint_position_targets.{name} must be numeric")
            numeric = float(target)
            if not math.isfinite(numeric):
                raise ConfigError(f"{prefix}.joint_position_targets.{name} must be finite")
            normalized_targets[name] = numeric
        normalized_commands.append(
            {"at_physics_step": step, "joint_position_targets": normalized_targets}
        )

    return {
        "physics_backend": backend,
        "experiment_id": experiment_id,
        "device": device,
        "renderer": renderer,
        "physics_dt_s": physics_dt,
        "expected_stage_meters_per_unit": meters_per_unit,
        "seed": seed,
        "scene_ref": scene_ref,
        "resolved_scene_ref": resolved_scene,
        "robot_asset_ref": asset_ref,
        "robot_prim_path": robot_prim,
        "expected_dof_names": dof_names,
        "joint_mapping_ref": joint_mapping_ref,
        "max_physics_steps": max_steps,
        "observation_every_physics_steps": observation_period,
        "warmup_physics_steps": warmup_physics_steps,
        "warmup_render_frames": warmup_render_frames,
        "cameras": cameras,
        "object_prim_path": object_prim,
        "contact_sensor_prim_path": contact_prim,
        "include_raw_contact": include_raw_contact,
        "video_fps": observation_rate_hz,
        "video_fourcc": video_fourcc,
        "commands": normalized_commands,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance(repo: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        status = command("status", "--porcelain")
        return {"commit": command("rev-parse", "HEAD"), "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    if hasattr(value, "cpu"):
        return _jsonable(value.cpu())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    try:
        return [_jsonable(item) for item in value]
    except TypeError:
        return str(value)


def _as_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
    return np.asarray(value)


def _camera_frame(value: Any, role: str, expected_hw: tuple[int, int]) -> Any:
    import numpy as np

    frame = _as_numpy(value)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(f"{role} RGB payload has unexpected shape {tuple(frame.shape)}")
    if tuple(frame.shape[:2]) != expected_hw:
        raise RuntimeError(
            f"{role} RGB shape {tuple(frame.shape[:2])} != configured {expected_hw}"
        )
    if frame.dtype != np.uint8:
        raise RuntimeError(f"{role} RGB dtype {frame.dtype} != uint8")
    return np.ascontiguousarray(frame)


def _single_articulation_vector(value: Any, label: str) -> Any:
    array = _as_numpy(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise RuntimeError(f"{label} has unexpected shape {tuple(array.shape)}")
    return array


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    json.dump(_jsonable(record), handle, ensure_ascii=False, allow_nan=False)
    handle.write("\n")
    handle.flush()


def _observed_backend(physics_scenes: list[Any]) -> dict[str, Any]:
    return {
        "scene_wrapper_types": [type(scene).__name__ for scene in physics_scenes],
        "scene_prim_paths": [str(scene.path) for scene in physics_scenes],
    }


def run_episode(
    config: dict[str, Any],
    runtime: dict[str, Any],
    config_path: Path,
    output_root: Path,
    episode_id: str,
    *,
    headless: bool,
) -> Path:
    """Start Isaac Sim, execute configured targets, and write one episode."""

    if not EPISODE_ID_RE.fullmatch(episode_id):
        raise ConfigError("episode id may contain only letters, digits, '.', '_' and '-'")
    episode_dir = output_root.resolve() / episode_id
    if episode_dir.exists():
        raise FileExistsError(f"episode output already exists: {episode_dir}")
    episode_dir.mkdir(parents=True)
    incomplete_marker = episode_dir / ".incomplete"
    incomplete_marker.write_text("episode has not finalized\n", encoding="utf-8")

    config_snapshot = episode_dir / "experiment_config.yaml"
    shutil.copyfile(config_path, config_snapshot)
    config_hash = _sha256(config_snapshot)
    repo_root = Path(__file__).resolve().parents[2]
    timeline_path = episode_dir / "timeline.jsonl"
    error_path = episode_dir / "run_error.txt"
    manifest_path = episode_dir / "episode_manifest.json"
    for role in ("base_rgb", "wrist_rgb"):
        (episode_dir / "streams" / role).mkdir(parents=True)
    (episode_dir / "videos").mkdir()

    manifest: dict[str, Any] = {
        "episode_schema_version": EPISODE_SCHEMA,
        "episode_id": episode_id,
        "experiment_id": runtime["experiment_id"],
        "experiment_config": {
            "storage_ref": config_snapshot.name,
            "sha256": config_hash,
            "schema_version": EXPERIMENT_SCHEMA,
        },
        "source": {
            "kind": "simulation",
            "simulator": {
                "name": "isaac_sim",
                "requested_version": ISAAC_VERSION,
                "physics_backend": runtime["physics_backend"],
                "device": runtime["device"],
                "renderer": runtime["renderer"],
                "physics_dt_s": runtime["physics_dt_s"],
                "seed": runtime["seed"],
                "scene_ref": runtime["scene_ref"],
            },
        },
        "robot": {
            "name": "piper_x",
            "asset_ref": runtime["robot_asset_ref"],
            "articulation_prim_path": runtime["robot_prim_path"],
            "joint_mapping_ref": runtime["joint_mapping_ref"],
        },
        "task": {
            "id": _at(config, "task.id"),
            "instruction": _at(config, "task.instruction"),
            "object_id": _at(config, "task.object_id"),
            "initial_condition_ref": _at(config, "task.initial_condition_ref"),
            "success_definition_ref": _at(config, "task.success_definition_ref"),
        },
        "provenance": {
            "repository": _git_provenance(repo_root),
            "python": sys.version,
            "headless": headless,
        },
        "clocks": {
            "simulation_time": {"unit": "s", "scope": "episode"},
            "physics_step": {"unit": "count", "scope": "Isaac Sim process"},
            "episode_physics_step": {"unit": "count", "scope": "episode after warmup"},
        },
        "streams": {},
        "action": {
            "source": "scripted_joint_position_targets",
            "semantics": _at(config, "execution.action_semantics"),
        },
        "termination": {"status": "failed", "reason": "runner did not finalize"},
        "result": {
            "task_success": None,
            "safety_evaluation_status": "not_evaluated",
            "safety_violations": None,
            "successful_but_unsafe": None,
        },
        "artifacts": {
            "timeline": timeline_path.name,
            "videos": {},
            "video_encoding": {
                "container": "mp4",
                "fps": runtime["video_fps"],
                "fourcc": runtime["video_fourcc"],
                "rgb_to_video_transform": "drop_alpha_if_present_then_rgb_to_bgr",
            },
            "error_log": None,
        },
        "integrity": {"observation_count": 0, "execution_count": 0},
    }
    if not runtime["resolved_scene_ref"].startswith(URI_PREFIXES):
        manifest["source"]["simulator"]["scene_sha256"] = _sha256(
            Path(runtime["resolved_scene_ref"])
        )

    simulation_app: Any = None
    app_utils: Any = None
    video_writers: dict[str, Any] = {}
    completed = False
    interrupted = False
    try:
        random.seed(runtime["seed"])
        import numpy as np

        np.random.seed(runtime["seed"])

        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {"headless": headless, "renderer": runtime["renderer"]}
        )

        import isaacsim.core.experimental.utils.app as app_utils_module

        app_utils = app_utils_module
        for extension in (
            "isaacsim.core.experimental.prims",
            "isaacsim.core.version",
            "isaacsim.core.rendering_manager",
            "isaacsim.core.simulation_manager",
            "isaacsim.sensors.experimental.physics",
            "isaacsim.sensors.experimental.rtx",
        ):
            enabled = app_utils.enable_extension(extension)
            if not enabled and not app_utils.is_extension_enabled(extension):
                raise RuntimeError(f"unable to enable Isaac Sim extension: {extension}")
            if not app_utils.is_extension_enabled(extension):
                raise RuntimeError(f"Isaac Sim extension is not enabled: {extension}")

        import cv2
        import yaml
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.version import get_version
        from isaacsim.sensors.experimental.physics import ContactSensor, JointStateSensor
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from pxr import UsdGeom

        manifest["provenance"]["dependencies"] = {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pyyaml": yaml.__version__,
        }

        version_parts = get_version()
        actual_version = ".".join(version_parts[2:5])
        manifest["source"]["simulator"]["actual_version_components"] = list(version_parts)
        if actual_version != ISAAC_VERSION:
            raise RuntimeError(
                f"Isaac Sim runtime {actual_version} does not match required {ISAAC_VERSION}"
            )

        opened, stage = stage_utils.open_stage(runtime["resolved_scene_ref"])
        if not opened or stage is None:
            raise RuntimeError(
                f"Isaac Sim could not open USD scene: {runtime['resolved_scene_ref']}"
            )

        actual_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        manifest["source"]["simulator"]["stage_meters_per_unit"] = actual_meters_per_unit
        if not math.isclose(
            actual_meters_per_unit,
            runtime["expected_stage_meters_per_unit"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "stage meters-per-unit "
                f"{actual_meters_per_unit} != configured "
                f"{runtime['expected_stage_meters_per_unit']}"
            )

        required_prims = {
            "robot articulation": runtime["robot_prim_path"],
            "target object": runtime["object_prim_path"],
            "contact sensor": runtime["contact_sensor_prim_path"],
            "base camera": runtime["cameras"]["base_rgb"]["prim_path"],
            "wrist camera": runtime["cameras"]["wrist_rgb"]["prim_path"],
        }
        for label, prim_path in required_prims.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"missing {label} prim: {prim_path}")
        for role in ("base_rgb", "wrist_rgb"):
            camera_prim = stage.GetPrimAtPath(runtime["cameras"][role]["prim_path"])
            if not camera_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"{role} prim is not a USD Camera: {camera_prim.GetPath()}")

        if not SimulationManager.switch_physics_engine(runtime["physics_backend"]):
            raise RuntimeError(
                f"unable to switch physics engine to {runtime['physics_backend']}"
            )
        SimulationManager.setup_simulation(
            dt=runtime["physics_dt_s"], device=runtime["device"]
        )
        RenderingManager.set_dt(runtime["physics_dt_s"])
        actual_device = str(SimulationManager.get_device())
        manifest["source"]["simulator"]["actual_device"] = actual_device
        if actual_device != runtime["device"]:
            raise RuntimeError(
                f"simulation device {actual_device} != configured {runtime['device']}"
            )
        physics_scenes = SimulationManager.get_physics_scenes()
        if not physics_scenes:
            raise RuntimeError("scene has no USD Physics Scene")
        manifest["source"]["simulator"]["physics_scene"] = _observed_backend(
            physics_scenes
        )
        actual_dts = [float(scene.get_dt()) for scene in physics_scenes]
        manifest["source"]["simulator"]["physics_scene"]["dt_s"] = actual_dts
        if any(
            not math.isclose(dt, runtime["physics_dt_s"], rel_tol=0.0, abs_tol=1e-12)
            for dt in actual_dts
        ):
            raise RuntimeError(
                f"physics scene dt {actual_dts} != configured {runtime['physics_dt_s']}"
            )

        articulation = Articulation(runtime["robot_prim_path"])
        joint_sensor = JointStateSensor(runtime["robot_prim_path"])
        target_object = RigidPrim(runtime["object_prim_path"])
        contact_sensor = ContactSensor(runtime["contact_sensor_prim_path"])
        if runtime["include_raw_contact"]:
            contact_sensor.add_raw_contact_data_to_frame()
        camera_sensors: dict[str, Any] = {}
        for role, camera_config in runtime["cameras"].items():
            camera = RtxCamera(
                camera_config["prim_path"], tick_rate=camera_config["tick_rate_hz"]
            )
            camera_sensors[role] = CameraSensor(
                camera,
                resolution=camera_config["resolution_hw"],
                annotators=["rgb"],
            )

        app_utils.play(commit=True)
        if runtime["warmup_physics_steps"]:
            SimulationManager.step(steps=runtime["warmup_physics_steps"])
        for _ in range(runtime["warmup_render_frames"]):
            RenderingManager.render()

        initial_joint_state = joint_sensor.get_data()
        if not initial_joint_state.get("is_valid", False):
            raise RuntimeError("JointStateSensor did not produce a valid warmup reading")
        actual_dof_names = list(initial_joint_state["dof_names"])
        if actual_dof_names != runtime["expected_dof_names"]:
            raise RuntimeError(
                f"actual DOF order {actual_dof_names} != configured "
                f"{runtime['expected_dof_names']}"
            )
        dof_types = _jsonable(initial_joint_state["dof_types"])
        manifest["robot"].update(
            {
                "dof_names": actual_dof_names,
                "dof_types": dof_types,
                "stage_meters_per_unit": initial_joint_state["stage_meters_per_unit"],
            }
        )
        lower_limits_raw, upper_limits_raw = articulation.get_dof_limits()
        lower_limits = _single_articulation_vector(lower_limits_raw, "DOF lower limits")
        upper_limits = _single_articulation_vector(upper_limits_raw, "DOF upper limits")
        stiffness_raw, damping_raw = articulation.get_dof_gains()
        stiffness = _single_articulation_vector(stiffness_raw, "DOF stiffness")
        damping = _single_articulation_vector(damping_raw, "DOF damping")
        drive_types_raw = articulation.get_dof_drive_types()
        drive_types = drive_types_raw[0] if len(drive_types_raw) == 1 else drive_types_raw
        if any(
            len(values) != len(actual_dof_names)
            for values in (lower_limits, upper_limits, stiffness, damping, drive_types)
        ):
            raise RuntimeError("articulation DOF metadata length does not match DOF names")
        manifest["robot"]["control"] = {
            "lower_limits": lower_limits.tolist(),
            "upper_limits": upper_limits.tolist(),
            "drive_types": _jsonable(drive_types),
            "stiffness": stiffness.tolist(),
            "damping": damping.tolist(),
        }

        dof_index_by_name = {name: index for index, name in enumerate(actual_dof_names)}
        controlled_names = sorted(
            {
                name
                for command in runtime["commands"]
                for name in command["joint_position_targets"]
            },
            key=dof_index_by_name.__getitem__,
        )
        for command in runtime["commands"]:
            for name, target in command["joint_position_targets"].items():
                index = dof_index_by_name[name]
                if target < float(lower_limits[index]) or target > float(upper_limits[index]):
                    raise RuntimeError(
                        f"target {target} for {name} is outside actual limits "
                        f"[{lower_limits[index]}, {upper_limits[index]}]"
                    )
                if str(drive_types[index]).lower() == "none":
                    raise RuntimeError(f"targeted DOF {name} has no drive in the loaded asset")
        articulation.switch_dof_control_mode(
            "position", dof_indices=articulation.get_dof_indices(controlled_names)
        )
        manifest["streams"]["robot_state"] = {
            "storage_ref": "timeline.jsonl#observation.robot_state",
            "dtype": "float",
            "shape": [len(actual_dof_names)],
            "fields": ["positions", "velocities", "efforts"],
            "dof_names": actual_dof_names,
            "dof_types": dof_types,
            "rotation_unit": "rad",
            "translation_unit": "m",
            "rotation_velocity_unit": "rad/s",
            "translation_velocity_unit": "m/s",
            "rotation_effort_unit": "N*m",
            "translation_effort_unit": "N",
            "clock": "simulation_time",
        }
        manifest["streams"]["object_state"] = {
            "storage_ref": "timeline.jsonl#observation.object_state",
            "prim_path": runtime["object_prim_path"],
            "position_unit": "stage_length_unit",
            "orientation_order": "wxyz",
            "linear_velocity_unit": "stage_length_unit/s",
            "angular_velocity_unit": "rad/s",
            "clock": "simulation_time",
        }
        manifest["streams"]["contact"] = {
            "storage_ref": "timeline.jsonl#observation.contact",
            "sensor_prim_path": runtime["contact_sensor_prim_path"],
            "raw_contact_included": runtime["include_raw_contact"],
            "force_unit": "kg*stage_length_unit/s^2",
            "raw_impulse_unit": None,
            "clock": "simulation_time",
        }
        manifest["streams"]["joint_effort"] = {
            "storage_ref": "timeline.jsonl#observation.robot_state.efforts",
            "shape": [len(actual_dof_names)],
            "dof_names": actual_dof_names,
            "dof_types": dof_types,
            "rotation_unit": "N*m",
            "translation_unit": "N",
            "clock": "simulation_time",
        }

        command_by_step = {
            command["at_physics_step"]: command for command in runtime["commands"]
        }
        observation_index = 0
        execution_count = 0
        with timeline_path.open("w", encoding="utf-8") as timeline:
            for episode_step in range(runtime["max_physics_steps"]):
                command = command_by_step.get(episode_step)
                if command is not None:
                    names = list(command["joint_position_targets"])
                    targets = np.asarray(
                        [command["joint_position_targets"][name] for name in names],
                        dtype=np.float32,
                    )
                    indices = articulation.get_dof_indices(names)
                    articulation.set_dof_position_targets(targets, dof_indices=indices)
                    _append_jsonl(
                        timeline,
                        {
                            "record_type": "execution",
                            "episode_physics_step": episode_step,
                            "isaac_physics_step_before_dispatch": SimulationManager.get_num_physics_steps(),
                            "simulation_time_s_before_dispatch": SimulationManager.get_simulation_time(),
                            "query_id": None,
                            "chunk_index": None,
                            "requested_action": {
                                "joint_names": names,
                                "joint_position_targets": targets.tolist(),
                            },
                            "executed_action": {
                                "joint_names": names,
                                "joint_position_targets": targets.tolist(),
                            },
                            "dispatch_status": "submitted_to_articulation_position_targets",
                        },
                    )
                    execution_count += 1

                SimulationManager.step(steps=1)
                should_observe = (
                    (episode_step + 1) % runtime["observation_every_physics_steps"] == 0
                )
                if not should_observe:
                    continue

                RenderingManager.render()
                joint_state = joint_sensor.get_data()
                if not joint_state.get("is_valid", False):
                    raise RuntimeError(
                        f"invalid joint state at episode physics step {episode_step}"
                    )
                contact_reading = contact_sensor.get_sensor_reading()
                if not contact_reading.is_valid:
                    raise RuntimeError(
                        f"invalid contact reading at episode physics step {episode_step}"
                    )
                contact = contact_sensor.get_data()
                contact["is_valid"] = bool(contact_reading.is_valid)
                object_positions, object_orientations = target_object.get_world_poses()
                object_linear_velocities, object_angular_velocities = target_object.get_velocities()

                image_refs: dict[str, str] = {}
                image_info: dict[str, Any] = {}
                for role, sensor in camera_sensors.items():
                    data, info = sensor.get_data("rgb")
                    if data is None:
                        raise RuntimeError(
                            f"{role} camera has no data after configured warmup"
                        )
                    frame = _camera_frame(
                        data, role, runtime["cameras"][role]["resolution_hw"]
                    )
                    relative_frame_path = (
                        Path("streams") / role / f"frame_{observation_index:06d}.npy"
                    )
                    np.save(episode_dir / relative_frame_path, frame, allow_pickle=False)
                    image_refs[role] = relative_frame_path.as_posix()
                    image_info[role] = _jsonable(info)

                    if role not in video_writers:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(episode_dir / "videos" / f"{role}.mp4"),
                            cv2.VideoWriter_fourcc(*runtime["video_fourcc"]),
                            runtime["video_fps"],
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"OpenCV could not open video writer for {role} with "
                                f"fourcc={runtime['video_fourcc']}"
                            )
                        video_writers[role] = writer
                        manifest["artifacts"]["videos"][role] = f"videos/{role}.mp4"
                        manifest["streams"][f"observation.image.{role}"] = {
                            "storage_ref": f"streams/{role}/*.npy",
                            "dtype": str(frame.dtype),
                            "shape": list(frame.shape),
                            "unit": None,
                            "frame": runtime["cameras"][role]["prim_path"],
                            "clock": "simulation_time",
                            "camera_tick_rate_hz": runtime["cameras"][role]["tick_rate_hz"],
                        }
                    video_frame = frame[:, :, :3]
                    video_writers[role].write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))

                _append_jsonl(
                    timeline,
                    {
                        "record_type": "observation",
                        "observation_index": observation_index,
                        "episode_physics_step": episode_step,
                        "isaac_physics_step": SimulationManager.get_num_physics_steps(),
                        "simulation_time_s": SimulationManager.get_simulation_time(),
                        "images": image_refs,
                        "image_info": image_info,
                        "robot_state": joint_state,
                        "object_state": {
                            "prim_path": runtime["object_prim_path"],
                            "positions": object_positions,
                            "orientations_wxyz": object_orientations,
                            "linear_velocities": object_linear_velocities,
                            "angular_velocities": object_angular_velocities,
                        },
                        "contact": contact,
                    },
                )
                observation_index += 1

        manifest["integrity"].update(
            {
                "observation_count": observation_index,
                "execution_count": execution_count,
                "configured_physics_steps": runtime["max_physics_steps"],
                "final_isaac_physics_step": SimulationManager.get_num_physics_steps(),
            }
        )
        manifest["termination"] = {
            "status": "complete",
            "reason": "configured_max_physics_steps_reached",
        }
        completed = True
    except KeyboardInterrupt:
        interrupted = True
        manifest["termination"] = {"status": "interrupted", "reason": "keyboard_interrupt"}
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        manifest["artifacts"]["error_log"] = error_path.name
        manifest["termination"] = {
            "status": "failed",
            "reason": traceback.format_exc().splitlines()[-1],
        }
        raise
    finally:
        for writer in video_writers.values():
            writer.release()
        if app_utils is not None:
            try:
                app_utils.stop()
            except Exception:
                pass
        if simulation_app is not None:
            simulation_app.close()
        _write_json(manifest_path, manifest)
        if completed or interrupted:
            incomplete_marker.unlink(missing_ok=True)

    if interrupted:
        raise KeyboardInterrupt
    return episode_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate config and referenced scene without starting Isaac Sim",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        print(f"config does not exist: {config_path}", file=sys.stderr)
        return 2
    try:
        config = _load_yaml(config_path)
        runtime = validate_config(config, config_path)
        if args.preflight:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "schema_version": EXPERIMENT_SCHEMA,
                        "scene_ref": runtime["resolved_scene_ref"],
                        "physics_backend": runtime["physics_backend"],
                        "command_count": len(runtime["commands"]),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.output_root is None or args.episode_id is None:
            raise ConfigError("--output-root and --episode-id are required unless --preflight is used")
        output = run_episode(
            config,
            runtime,
            config_path,
            args.output_root,
            args.episode_id,
            headless=args.headless,
        )
        print(output)
        return 0
    except KeyboardInterrupt:
        print("interrupted: episode manifest was finalized", file=sys.stderr)
        return 130
    except (ConfigError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
