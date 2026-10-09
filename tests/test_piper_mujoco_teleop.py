from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fragilegrab.benchmark.teleop import (
    LatestLeaderReader,
    PositionMapper,
    latency_summary_ms,
    load_teleop_config,
)


ROOT = Path(__file__).resolve().parents[1]


def mapping_config(alpha: float = 1.0) -> dict:
    return {
        "joints": {
            "sign": [1, -1, 1, -1, 1, -1],
            "scale": [2, 2, 2, 2, 2, 2],
            "offset_rad": [0.1] * 6,
            "limits_rad": [[-1, 1]] * 6,
        },
        "gripper": {
            "input_closedness_limits": [0, 1],
            "sim_open_m": 0.05,
            "sim_closed_m": 0.0,
            "limits_m": [0.0, 0.05],
        },
        "smoothing": {"enabled": alpha != 1.0, "alpha": alpha},
    }


def test_mapping_is_sign_scale_offset_clamp_then_filter() -> None:
    mapper = PositionMapper(mapping_config(alpha=0.5), np.array([[-1, 1]] * 6 + [[0, 0.05]]))
    first, audit = mapper.map(np.array([1, 1, 1, 1, 1, 1, 0]))
    np.testing.assert_allclose(first, [1, -1, 1, -1, 1, -1, 0.05])
    assert audit["unclamped_target"][:2] == [2.1, -1.9]
    second, _ = mapper.map(np.zeros(7))
    np.testing.assert_allclose(second, [0.55, -0.45, 0.55, -0.45, 0.55, -0.45, 0.05])


def test_gripper_records_actual_sim_actuator_target() -> None:
    mapper = PositionMapper(mapping_config())
    opened, _ = mapper.map(np.zeros(7))
    closed, _ = mapper.map(np.r_[np.zeros(6), 1.0])
    assert opened[6] == 0.05
    assert closed[6] == 0.0


class FakeAdapter:
    def __init__(self) -> None:
        self.closed = False

    def start(self) -> dict:
        return {"fake": True}

    def read_state(self):
        arm = SimpleNamespace(state=np.arange(7), raw={"source": "fake"})
        return SimpleNamespace(value=arm, timestamp_ns=time.monotonic_ns())

    def close(self) -> None:
        self.closed = True


def test_reader_runs_independently_and_closes() -> None:
    adapter = FakeAdapter()
    reader = LatestLeaderReader(adapter, 100)
    reader.start()
    sample = reader.latest(timeout_s=0.2)
    np.testing.assert_array_equal(sample.state, np.arange(7))
    reader.close()
    assert adapter.closed


def test_latency_summary_reports_mean_and_p95() -> None:
    result = latency_summary_ms([10_000_000, 20_000_000, 30_000_000])
    assert result["samples"] == 3
    assert result["mean_ms"] == 20.0
    assert result["p95_ms"] == 29.0


def test_versioned_teleop_config_separates_record_and_preview_resolution() -> None:
    config = load_teleop_config(ROOT / "configs/teleop/piper_mujoco.yaml")
    assert config["version"] == "piper_mujoco_teleop/0.2"
    assert config["visual"]["record_resolution_hw"] == [480, 640]
    assert config["visual"]["preview_resolution_hw"] == [240, 320]
    assert config["loops"]["render_hz"] == 20.0
