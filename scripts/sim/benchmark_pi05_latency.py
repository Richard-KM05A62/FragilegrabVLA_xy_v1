#!/usr/bin/env python3
"""对固定 MuJoCo observation 连续调用 π0.5，分解模型、服务端和端到端延迟；不执行动作。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.environment import MujocoQualificationEnv, load_configs
from fragilegrab.benchmark.pi05_client import Pi05WebsocketClient
from fragilegrab.benchmark.pi05_contract import Pi05ContractError, PiperPi05Contract
from fragilegrab.benchmark.pi05_runner import load_rollout_config
from fragilegrab.benchmark.runner import native


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--benchmark-config-dir", type=Path, default=ROOT / "configs/benchmark")
    parser.add_argument("--rollout-config", type=Path, required=True)
    parser.add_argument("--policy-host", required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="可选 JSON 报告；拒绝覆盖")
    args = parser.parse_args()
    robot_assets = args.robot_assets
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup 必须非负，repeats 必须为正整数")

    docs = load_configs(args.benchmark_config_dir)
    rollout = load_rollout_config(args.rollout_config)
    seed = docs["benchmark"]["values"]["seed"] if args.seed is None else args.seed
    contract = PiperPi05Contract(rollout["norm_stats_id"])

    env = MujocoQualificationEnv(docs, robot_assets)
    try:
        observation = env.reset(seed)
        request = contract.encode_observation(observation, rollout["prompt"])
        with Pi05WebsocketClient(args.policy_host, args.policy_port) as policy:
            contract.validate_server_metadata(policy.metadata)
            rows = []
            for index in range(args.warmup + args.repeats):
                started = time.monotonic_ns()
                response = policy.infer(request)
                round_trip_ms = (time.monotonic_ns() - started) / 1e6
                actions = np.asarray(response.get("actions"))
                action_validation_error = None
                try:
                    contract.decode_action_chunk(response, env.model.actuator_ctrlrange)
                except Pi05ContractError as exc:
                    action_validation_error = str(exc)
                rows.append(
                    {
                        "index": index,
                        "phase": "warmup" if index < args.warmup else "measured",
                        "round_trip_ms": round_trip_ms,
                        "server_infer_ms": (response.get("server_timing") or {}).get("infer_ms"),
                        "server_prev_total_ms": (response.get("server_timing") or {}).get("prev_total_ms"),
                        "model_infer_ms": (response.get("policy_timing") or {}).get("infer_ms"),
                        "action_shape": list(actions.shape),
                        "gripper_closure_min": (
                            float(actions[:, 6].min()) if actions.shape == (contract.action_horizon, 7) else None
                        ),
                        "gripper_closure_max": (
                            float(actions[:, 6].max()) if actions.shape == (contract.action_horizon, 7) else None
                        ),
                        "action_validation_error": action_validation_error,
                    }
                )
    finally:
        env.close()

    measured = [row for row in rows if row["phase"] == "measured"]
    report = {
        "schema_version": "fragilegrab.pi05_latency/0.1",
        "status": "measured_without_action_execution",
        "warmup_count": args.warmup,
        "measurement_count": args.repeats,
        "invalid_action_chunks": sum(row["action_validation_error"] is not None for row in rows),
        "rows": rows,
        "summary_ms": {
            key: _summary([float(row[key]) for row in measured if row[key] is not None])
            for key in ("round_trip_ms", "server_infer_ms", "model_infer_ms")
            if any(row[key] is not None for row in measured)
        },
    }
    rendered = json.dumps(native(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"拒绝覆盖：{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
