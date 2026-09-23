#!/usr/bin/env python3
"""从已完成 MuJoCo episode 生成 PiPER π0.5 接口 normalization stats。"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fragilegrab.benchmark.pi05_contract import (
    CONTRACT_ID,
    OPENPI_COMMIT,
    OPENPI_INTERFACE_BASE_COMMIT,
    PiperPi05Contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _episode_directories(inputs: list[Path]) -> list[Path]:
    episodes: dict[str, Path] = {}
    for value in inputs:
        value = value.resolve()
        manifests = [value / "episode_manifest.json"] if (value / "episode_manifest.json").exists() else sorted(value.rglob("episode_manifest.json"))
        for manifest_path in manifests:
            manifest = _read_json(manifest_path)
            if (manifest.get("termination") or {}).get("status") != "complete":
                raise ValueError(f"episode 未完整完成：{manifest_path}")
            if (manifest.get("robot") or {}).get("name") != "piper_x":
                raise ValueError(f"拒绝使用非 PiPER-X episode 计算 normalization：{manifest_path}")
            timeline = manifest_path.parent / manifest.get("artifacts", {}).get("timeline", "timeline.jsonl")
            if not timeline.is_file():
                raise ValueError(f"缺少 timeline：{timeline}")
            identity = (manifest.get("integrity") or {}).get("timeline_sha256") or _sha256(timeline)
            episodes.setdefault(identity, manifest_path.parent)
    if not episodes:
        raise ValueError("没有找到完整 episode")
    return list(episodes.values())


def _samples(episode: Path, contract: PiperPi05Contract) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = _read_json(episode / "episode_manifest.json")
    timeline = episode / manifest.get("artifacts", {}).get("timeline", "timeline.jsonl")
    executions: list[tuple[float, np.ndarray]] = []
    observations: list[tuple[float, np.ndarray]] = []
    timeline_open = gzip.open if timeline.suffix == ".gz" else open
    with timeline_open(timeline, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            record_type = row.get("record_type")
            if record_type == "execution" and "requested_action" in row:
                executions.append((float(row["simulation_time_s"]), contract.policy_action(row["requested_action"])))
            elif record_type == "observation" and "joint_position" in row:
                observations.append((float(row["simulation_time_s"]), contract.policy_state(row["joint_position"])))
    if len(executions) < 2 or len(observations) < 2:
        raise ValueError(f"episode 样本不足：{episode}")
    execution_times = [time_s for time_s, _ in executions]
    states, chunks = [], []
    for time_s, state in observations:
        index = bisect.bisect_right(execution_times, time_s) - 1
        if index < 0:
            continue
        future = [action for _, action in executions[index : index + contract.action_horizon]]
        if not future:
            continue
        future.extend([future[-1]] * (contract.action_horizon - len(future)))
        chunk = np.stack(future)
        chunk[:, :6] -= state[:6]
        states.append(state)
        chunks.append(chunk)
    if len(states) < 2:
        raise ValueError(f"episode 对齐后样本不足：{episode}")
    evidence = {
        "episode_id": manifest.get("episode_id"),
        "directory": str(episode),
        "timeline_sha256": _sha256(timeline),
        "observations_used": len(states),
        "executions": len(executions),
    }
    return np.stack(states), np.stack(chunks), evidence


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    return {
        "mean": np.mean(flat, axis=0).tolist(),
        "std": np.std(flat, axis=0).tolist(),
        "q01": np.quantile(flat, 0.01, axis=0).tolist(),
        "q99": np.quantile(flat, 0.99, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="episode 或包含 episode 的 run 目录")
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的输出目录")
    parser.add_argument("--norm-stats-id", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{args.output}")

    contract = PiperPi05Contract(args.norm_stats_id)
    episode_dirs = _episode_directories(args.inputs)
    state_batches, action_batches, evidence = [], [], []
    for episode in episode_dirs:
        states, actions, item = _samples(episode, contract)
        state_batches.append(states)
        action_batches.append(actions)
        evidence.append(item)
    states = np.concatenate(state_batches)
    actions = np.concatenate(action_batches)

    args.output.mkdir(parents=True)
    norm_stats = {"norm_stats": {"state": _stats(states), "actions": _stats(actions)}}
    stats_path = args.output / "norm_stats.json"
    stats_path.write_text(json.dumps(norm_stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    provenance = {
        "schema_version": "fragilegrab.pi05_norm_stats/0.1",
        "norm_stats_id": args.norm_stats_id,
        "contract_id": CONTRACT_ID,
        "robot_model": "piper_x",
        "purpose": "piper_x_dataset_calibration",
        "execution_permitted": True,
        "openpi_commit": OPENPI_COMMIT,
        "openpi_interface_base_commit": OPENPI_INTERFACE_BASE_COMMIT,
        "state_semantics": "joint1..joint6 rad + gripper closure [0,1]",
        "action_pre_normalization": "joint1..joint6 delta from current state + absolute gripper closure [0,1]",
        "action_horizon": contract.action_horizon,
        "deduplicated_episode_count": len(episode_dirs),
        "state_sample_count": int(states.shape[0]),
        "action_vector_count": int(np.prod(actions.shape[:-1])),
        "episodes": evidence,
        "norm_stats_sha256": _sha256(stats_path),
        "limitation": "来自脚本 MuJoCo 轨迹的接口校准统计；不是 π0.5 微调，也不证明真实 Piper 分布。",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
