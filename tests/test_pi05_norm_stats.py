from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compute_piper_norm_stats", ROOT / "scripts/policy/compute_piper_norm_stats.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _episode(root: Path, name: str, offset: float = 0.0) -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    timeline = episode / "timeline.jsonl"
    rows = []
    for index in range(4):
        time_s = index * 0.05
        action = [offset + index * 0.01, 1.0, -1.0, 0.0, 0.0, 0.0, 0.05 if index < 2 else 0.0]
        state = [offset + index * 0.005, 1.0, -1.0, 0.0, 0.0, 0.0, action[6], -action[6]]
        rows.append({"record_type": "execution", "simulation_time_s": time_s, "requested_action": action})
        rows.append({"record_type": "observation", "simulation_time_s": time_s + 0.001, "joint_position": state})
    timeline.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "episode_id": name,
        "robot": {"name": "piper_x"},
        "termination": {"status": "complete"},
        "artifacts": {"timeline": "timeline.jsonl"},
        "integrity": {},
    }
    (episode / "episode_manifest.json").write_text(json.dumps(manifest))
    return episode


def test_stats_samples_use_delta_joints_absolute_gripper_and_deduplicate(tmp_path: Path) -> None:
    episode = _episode(tmp_path, "episode")
    contract = MODULE.PiperPi05Contract("piper_x_test")
    states, actions, evidence = MODULE._samples(episode, contract)
    assert states.shape == (4, 7)
    assert actions.shape == (4, 50, 7)
    assert states[0, 6] == 0.0
    assert states[-1, 6] == 1.0
    assert actions[0, 0, 0] == 0.0
    assert actions[0, 1, 0] == 0.01
    assert actions[0, 2, 6] == 1.0
    assert evidence["observations_used"] == 4

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "timeline.jsonl").write_bytes((episode / "timeline.jsonl").read_bytes())
    (duplicate / "episode_manifest.json").write_text(
        (episode / "episode_manifest.json").read_text()
    )
    found = MODULE._episode_directories([tmp_path])
    assert len(found) == 1


def test_stats_json_has_openpi_compatible_keys(tmp_path: Path) -> None:
    values = np.array([[0.0, 1.0], [2.0, 3.0]])
    stats = MODULE._stats(values)
    assert set(stats) == {"mean", "std", "q01", "q99"}
    np.testing.assert_allclose(stats["mean"], [1.0, 2.0])
