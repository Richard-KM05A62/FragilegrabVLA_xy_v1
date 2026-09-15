from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_module("run_instrumented_episode", "scripts/sim/run_instrumented_episode.py")
episode_validator = _load_module("validate_episode", "scripts/sim/validate_episode.py")


def _valid_config(scene_ref: str) -> dict:
    return {
        "schema_version": "fragilegrab.experiment/0.1",
        "experiment": {"id": "test"},
        "source": {
            "kind": "simulation",
            "simulator": {
                "name": "isaac_sim",
                "version": "6.1.0",
                "physics_backend": "physx",
                "device": "cpu",
                "physics_dt_s": 0.01,
                "renderer": "RaytracedLighting",
                "expected_stage_meters_per_unit": 1.0,
                "scene_ref": scene_ref,
                "robot_asset_ref": "asset-manifest:test",
                "seed": 7,
            },
        },
        "robot": {
            "name": "piper_x",
            "articulation_prim_path": "/World/PiperX",
            "expected_dof_names": ["joint1", "gripper_joint1"],
            "joint_mapping_ref": "test",
        },
        "task": {
            "id": "simple_grasp",
            "instruction": "test",
            "object_id": "object",
            "initial_condition_ref": "test",
            "success_definition_ref": "TBD",
        },
        "policy": {"enabled": False, "name": "pi05"},
        "execution": {
            "action_source": "scripted_joint_position_targets",
            "action_semantics": {
                "mode": "position",
                "value_type": "absolute",
                "rotation_unit": "rad",
                "translation_unit": "m",
            },
            "scripted_commands": [
                {
                    "at_physics_step": 0,
                    "joint_position_targets": {"joint1": 0.1},
                }
            ],
        },
        "recording": {
            "episode_schema_version": "fragilegrab.episode/0.1",
            "preserve_per_source_timing": True,
            "required_streams": [
                "observation.image.base_rgb",
                "observation.image.wrist_rgb",
                "robot_state",
                "object_state",
                "contact",
                "joint_effort",
            ],
            "max_physics_steps": 2,
            "observation_every_physics_steps": 1,
            "warmup_physics_steps": 1,
            "warmup_render_frames": 1,
            "cameras": {
                "base_rgb": {
                    "prim_path": "/World/BaseCamera",
                    "tick_rate_hz": 100,
                    "resolution_hw": [2, 3],
                },
                "wrist_rgb": {
                    "prim_path": "/World/PiperX/WristCamera",
                    "tick_rate_hz": 100,
                    "resolution_hw": [2, 3],
                },
            },
            "object_state": {"prim_path": "/World/Object"},
            "contact": {
                "sensor_prim_path": "/World/Object/ContactSensor",
                "include_raw": True,
            },
            "video": {"enabled": True, "fourcc": "mp4v"},
        },
    }


def _write_video(path: Path, frame: np.ndarray, fps: float = 100.0) -> None:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"test video writer could not open {path}")
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _write_timeline(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_complete_episode(root: Path) -> tuple[dict, list[dict]]:
    config = root / "experiment_config.yaml"
    config.write_text("schema_version: fragilegrab.experiment/0.1\n", encoding="utf-8")
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    original_config = root / "experiment_config.original.yaml"
    original_config.write_bytes(config.read_bytes())
    original_config_hash = hashlib.sha256(original_config.read_bytes()).hexdigest()
    (root / "streams" / "base_rgb").mkdir(parents=True)
    (root / "streams" / "wrist_rgb").mkdir(parents=True)
    (root / "videos").mkdir()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    np.save(root / "streams" / "base_rgb" / "frame_000000.npy", frame)
    np.save(root / "streams" / "wrist_rgb" / "frame_000000.npy", frame)
    _write_video(root / "videos" / "base_rgb.mp4", frame)
    _write_video(root / "videos" / "wrist_rgb.mp4", frame)

    records = [
        {
            "record_type": "execution",
            "requested_action": {
                "joint_names": ["joint1"],
                "joint_position_targets": [0.1],
            },
            "executed_action": {
                "joint_names": ["joint1"],
                "joint_position_targets": [0.1],
            },
            "dispatch_status": "submitted_to_articulation_position_targets",
        },
        {
            "record_type": "observation",
            "episode_physics_step": 0,
            "simulation_time_s": 0.01,
            "images": {
                "base_rgb": "streams/base_rgb/frame_000000.npy",
                "wrist_rgb": "streams/wrist_rgb/frame_000000.npy",
            },
            "robot_state": {
                "is_valid": True,
                "dof_names": ["joint1", "gripper_joint1"],
                "positions": [0.0, 0.0],
                "velocities": [0.0, 0.0],
                "efforts": [0.0, 0.0],
                "dof_types": [0, 1],
            },
            "object_state": {"positions": [[0.0, 0.0, 0.0]]},
            "contact": {"is_valid": True, "in_contact": False},
        },
    ]
    _write_timeline(root / "timeline.jsonl", records)

    image_stream = {"dtype": "uint8", "shape": [16, 16, 3]}
    dof_stream = {
        "dtype": "float",
        "shape": [2],
        "dof_names": ["joint1", "gripper_joint1"],
        "dof_types": [0, 1],
    }
    manifest = {
        "episode_schema_version": "fragilegrab.episode/0.1",
        "episode_id": "test",
        "experiment_config": {
            "storage_ref": config.name,
            "sha256": config_hash,
            "source_snapshot": {
                "storage_ref": original_config.name,
                "sha256": original_config_hash,
            },
        },
        "source": {"kind": "simulation"},
        "action": {"source": "scripted_joint_position_targets"},
        "termination": {"status": "complete"},
        "streams": {
            "observation.image.base_rgb": image_stream,
            "observation.image.wrist_rgb": image_stream,
            "robot_state": dof_stream,
            "object_state": {},
            "contact": {},
            "joint_effort": copy.deepcopy(dof_stream),
        },
        "artifacts": {
            "timeline": "timeline.jsonl",
            "videos": {
                "base_rgb": "videos/base_rgb.mp4",
                "wrist_rgb": "videos/wrist_rgb.mp4",
            },
            "video_encoding": {"container": "mp4", "fps": 100.0},
        },
        "integrity": {"observation_count": 1, "execution_count": 1},
    }
    (root / "episode_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest, records


class ConfigContractTest(unittest.TestCase):
    def test_valid_policy_free_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene.usd"
            scene.write_bytes(b"test")
            config_path = root / "config.yaml"
            runtime = runner.validate_config(_valid_config(scene.name), config_path)
            self.assertEqual(runtime["resolved_scene_ref"], str(scene))
            self.assertEqual(runtime["commands"][0]["joint_position_targets"], {"joint1": 0.1})

    def test_tbd_runtime_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene.usd"
            scene.write_bytes(b"test")
            config = _valid_config(scene.name)
            config["source"]["simulator"]["physics_backend"] = "TBD"
            with self.assertRaises(runner.ConfigError):
                runner.validate_config(config, root / "config.yaml")

    def test_duplicate_camera_prim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene.usd"
            scene.write_bytes(b"test")
            config = _valid_config(scene.name)
            config["recording"]["cameras"]["wrist_rgb"]["prim_path"] = (
                config["recording"]["cameras"]["base_rgb"]["prim_path"]
            )
            with self.assertRaises(runner.ConfigError):
                runner.validate_config(config, root / "config.yaml")

    def test_seed_outside_numpy_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene.usd"
            scene.write_bytes(b"test")
            for seed in (-1, 2**32):
                with self.subTest(seed=seed):
                    config = _valid_config(scene.name)
                    config["source"]["simulator"]["seed"] = seed
                    with self.assertRaises(runner.ConfigError):
                        runner.validate_config(config, root / "config.yaml")

    def test_missing_success_definition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene.usd"
            scene.write_bytes(b"test")
            config = _valid_config(scene.name)
            del config["task"]["success_definition_ref"]
            with self.assertRaises(runner.ConfigError):
                runner.validate_config(config, root / "config.yaml")

    def test_config_snapshot_normalizes_relative_scene_ref(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            scene = input_dir / "scene.usd"
            scene.write_bytes(b"test")
            config = _valid_config(scene.name)
            config_path = input_dir / "config.yaml"
            runner._write_yaml(config_path, config)
            runtime = runner.validate_config(config, config_path)
            episode_dir = root / "episode"
            episode_dir.mkdir()

            descriptor = runner._snapshot_configs(
                config, runtime, config_path, episode_dir
            )

            original_snapshot = episode_dir / "experiment_config.original.yaml"
            effective_snapshot = episode_dir / "experiment_config.yaml"
            effective_config = runner._load_yaml(effective_snapshot)
            self.assertEqual(original_snapshot.read_bytes(), config_path.read_bytes())
            self.assertEqual(
                effective_config["source"]["simulator"]["scene_ref"], str(scene)
            )
            self.assertEqual(
                descriptor["normalized_fields"]["source.simulator.scene_ref"],
                {"input": "scene.usd", "effective": str(scene)},
            )
            self.assertEqual(
                descriptor["sha256"],
                hashlib.sha256(effective_snapshot.read_bytes()).hexdigest(),
            )


class EpisodeContractTest(unittest.TestCase):
    def test_complete_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_complete_episode(root)

            result = episode_validator.validate_episode(root)
            self.assertEqual(result["observations"], 1)
            self.assertEqual(result["executions"], 1)

    def test_corrupt_video_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_complete_episode(root)
            (root / "videos" / "base_rgb.mp4").write_bytes(b"video")

            with self.assertRaises(episode_validator.EpisodeError):
                episode_validator.validate_episode(root)

    def test_robot_state_dof_order_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records = _write_complete_episode(root)
            records[1]["robot_state"]["dof_names"] = [
                "gripper_joint1",
                "joint1",
            ]
            _write_timeline(root / "timeline.jsonl", records)

            with self.assertRaises(episode_validator.EpisodeError):
                episode_validator.validate_episode(root)

    def test_joint_effort_dof_contract_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = _write_complete_episode(root)
            manifest["streams"]["joint_effort"]["dof_types"] = [1, 0]
            (root / "episode_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaises(episode_validator.EpisodeError):
                episode_validator.validate_episode(root)


if __name__ == "__main__":
    unittest.main()
