#!/usr/bin/env python3
"""生成仅用于验证 π0.5 数据流的合成 normalization，禁止动作执行。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fragilegrab.benchmark.pi05_contract import (  # noqa: E402
    CONTRACT_ID,
    OPENPI_COMMIT,
    OPENPI_INTERFACE_BASE_COMMIT,
    PiperPi05Contract,
)


DEFAULT_NORM_STATS_ID = "piper_x_workflow_smoke_v0.1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_smoke_stats(output: Path, norm_stats_id: str) -> dict:
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    contract = PiperPi05Contract(norm_stats_id)
    identity = {
        "mean": [0.0] * 7,
        "std": [1.0] * 7,
        "q01": [-1.0] * 7,
        "q99": [1.0] * 7,
    }
    output.mkdir(parents=True)
    stats_path = output / "norm_stats.json"
    stats_path.write_text(
        json.dumps(
            {"norm_stats": {"state": identity, "actions": identity}},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "fragilegrab.pi05_norm_stats/0.1",
        "norm_stats_id": norm_stats_id,
        "contract_id": CONTRACT_ID,
        "robot_model": "piper_x",
        "openpi_commit": OPENPI_COMMIT,
        "openpi_interface_base_commit": OPENPI_INTERFACE_BASE_COMMIT,
        "purpose": "transport_only",
        "execution_permitted": False,
        "state_semantics": "joint1..joint6 rad + gripper closure [0,1]",
        "action_pre_normalization": "joint1..joint6 delta from current state + absolute gripper closure [0,1]",
        "action_horizon": contract.action_horizon,
        "source": "synthetic identity-like scale; not derived from robot or task data",
        "norm_stats_sha256": _sha256(stats_path),
        "limitation": "仅验证序列化、网络传输和云端推理；禁止执行输出，不代表 PiPER-X 能力或有效 calibration。",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的输出目录")
    parser.add_argument("--norm-stats-id", default=DEFAULT_NORM_STATS_ID)
    args = parser.parse_args()
    print(json.dumps(create_smoke_stats(args.output, args.norm_stats_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
