from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

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


class EpisodeContractTest(unittest.TestCase):
    def test_complete_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiment_config.yaml"
            config.write_text("schema_version: fragilegrab.experiment/0.1\n", encoding="utf-8")
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            (root / "streams" / "base_rgb").mkdir(parents=True)
            (root / "streams" / "wrist_rgb").mkdir(parents=True)
            (root / "videos").mkdir()
            frame = np.zeros((2, 3, 3), dtype=np.uint8)
            np.save(root / "streams" / "base_rgb" / "frame_000000.npy", frame)
            np.save(root / "streams" / "wrist_rgb" / "frame_000000.npy", frame)
            (root / "videos" / "base_rgb.mp4").write_bytes(b"video")
            (root / "videos" / "wrist_rgb.mp4").write_bytes(b"video")

            records = [
                {
                    "record_type": "execution",
                    "requested_action": {"joint_names": ["joint1"], "joint_position_targets": [0.1]},
                    "executed_action": {"joint_names": ["joint1"], "joint_position_targets": [0.1]},
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
                        "dof_names": ["joint1"],
                        "positions": [0.0],
                        "velocities": [0.0],
                        "efforts": [0.0],
                        "dof_types": [0],
                    },
                    "object_state": {"positions": [[0.0, 0.0, 0.0]]},
                    "contact": {"is_valid": True, "in_contact": False},
                },
            ]
            with (root / "timeline.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            stream_descriptor = {"dtype": "uint8", "shape": [2, 3, 3]}
            manifest = {
                "episode_schema_version": "fragilegrab.episode/0.1",
                "episode_id": "test",
                "experiment_config": {
                    "storage_ref": config.name,
                    "sha256": config_hash,
                },
                "source": {"kind": "simulation"},
                "action": {"source": "scripted_joint_position_targets"},
                "termination": {"status": "complete"},
                "streams": {
                    "observation.image.base_rgb": stream_descriptor,
                    "observation.image.wrist_rgb": stream_descriptor,
                    "robot_state": {},
                    "object_state": {},
                    "contact": {},
                    "joint_effort": {},
                },
                "artifacts": {
                    "timeline": "timeline.jsonl",
                    "videos": {
                        "base_rgb": "videos/base_rgb.mp4",
                        "wrist_rgb": "videos/wrist_rgb.mp4",
                    },
                },
                "integrity": {"observation_count": 1, "execution_count": 1},
            }
            (root / "episode_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = episode_validator.validate_episode(root)
            self.assertEqual(result["observations"], 1)
            self.assertEqual(result["executions"], 1)


if __name__ == "__main__":
    unittest.main()
