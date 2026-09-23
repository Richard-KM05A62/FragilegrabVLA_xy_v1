#!/usr/bin/env python3
"""在已验证可执行 normalization 上运行小规模 execution-horizon sweep。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.pi05_client import Pi05WebsocketClient  # noqa: E402
from fragilegrab.benchmark.pi05_contract import PiperPi05Contract  # noqa: E402
from fragilegrab.benchmark.pi05_runner import load_rollout_config, run_pi05_episode  # noqa: E402
from fragilegrab.benchmark.runner import save_json  # noqa: E402


def _load_sweep(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "fragilegrab.pi05_horizon_sweep/0.1":
        raise ValueError("horizon sweep schema_version 不符")
    horizons = value.get("execution_horizons")
    seeds = value.get("seeds")
    if not isinstance(horizons, list) or not horizons or any(type(x) is not int or x <= 0 for x in horizons):
        raise ValueError("execution_horizons 必须是非空正整数列表")
    if len(set(horizons)) != len(horizons) or any(x > 50 for x in horizons):
        raise ValueError("execution_horizons 必须唯一且不超过 action horizon 50")
    if not isinstance(seeds, list) or not seeds or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("seeds 必须是非空 uint32 整数列表")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--benchmark-config-dir", type=Path, default=ROOT / "configs/benchmark")
    parser.add_argument("--object-config", type=Path, help="可选 project/LIBERO 场景配置")
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-host", required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖：{args.output}")

    sweep = _load_sweep(args.sweep_config)
    base_path = ROOT / sweep["base_rollout_config"]
    base = load_rollout_config(base_path)
    contract = PiperPi05Contract(base["norm_stats_id"])
    args.output.mkdir(parents=True)
    results = []
    with Pi05WebsocketClient(args.policy_host, args.policy_port) as policy:
        contract.validate_server_metadata(policy.metadata, require_execution=True)
        for horizon in sweep["execution_horizons"]:
            for seed in sweep["seeds"]:
                effective = {**base, "execution_horizon": horizon}
                config_path = args.output / f"horizon_{horizon}_seed_{seed}.yaml"
                config_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
                episode_dir = args.output / f"horizon_{horizon}_seed_{seed}"
                manifest = run_pi05_episode(
                    args.benchmark_config_dir,
                    config_path,
                    args.robot_assets,
                    episode_dir,
                    policy,
                    seed=seed,
                    object_config=args.object_config,
                )
                metrics = manifest["result"]
                tendency = metrics["grasp_tendency"]
                results.append(
                    {
                        "execution_horizon": horizon,
                        "seed": seed,
                        "task_success": metrics["task_success"],
                        "qualification": metrics["qualification"],
                        "slip": metrics["slip"],
                        "drop": metrics["drop"],
                        "unintended_collision_events": metrics["unintended_collision_events"],
                        "max_lift_m": metrics["max_lift_m"],
                        "highest_grasp_stage": tendency["highest_stage"],
                        "highest_grasp_stage_label": tendency["highest_stage_label"],
                        "initial_ee_object_distance_m": tendency["initial_ee_object_distance_m"],
                        "minimum_ee_object_distance_m": tendency["minimum_ee_object_distance_m"],
                        "gripper_close_time_s": tendency["gripper_close_time_s"],
                        "gripper_close_ee_object_distance_m": tendency[
                            "gripper_close_ee_object_distance_m"
                        ],
                        "finger_object_contact": tendency["finger_object_contact"],
                        "bilateral_finger_contact": tendency["bilateral_finger_contact"],
                        "lift_attempt": tendency["lift_attempt"],
                        "object_maximum_lift_m": tendency["object_maximum_lift_m"],
                        "query_count": metrics["policy_rollout"]["query_count"],
                        "actions_executed": metrics["policy_rollout"]["actions_executed"],
                        "termination": metrics["policy_rollout"]["termination"],
                        "episode_duration_s": (
                            metrics["policy_rollout"]["physics_steps"]
                            / (
                                manifest["execution"]["control_rate_hz"]
                                * manifest["execution"]["control_steps_per_action"]
                            )
                        ),
                    }
                )
    by_horizon = {}
    for horizon in sweep["execution_horizons"]:
        rows = [row for row in results if row["execution_horizon"] == horizon]
        counts = Counter(row["highest_grasp_stage"] for row in rows)
        by_horizon[str(horizon)] = {
            "episode_count": len(rows),
            "highest_stage_counts": {f"L{stage}": counts.get(stage, 0) for stage in range(7)},
            "reached_or_exceeded": {
                f"L{stage}": sum(row["highest_grasp_stage"] >= stage for row in rows)
                for stage in range(1, 7)
            },
            "minimum_ee_object_distance_m": [row["minimum_ee_object_distance_m"] for row in rows],
            "gripper_close_ee_object_distance_m": [
                row["gripper_close_ee_object_distance_m"] for row in rows
                if row["gripper_close_ee_object_distance_m"] is not None
            ],
            "object_maximum_lift_m": [row["object_maximum_lift_m"] for row in rows],
        }
    summary = {
        "schema_version": "fragilegrab.pi05_horizon_sweep_result/0.1",
        "base_rollout_config": str(base_path.relative_to(ROOT)),
        "policy_metadata": policy.metadata,
        "results": results,
        "by_horizon": by_horizon,
    }
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
