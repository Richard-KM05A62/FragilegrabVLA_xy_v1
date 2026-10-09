#!/usr/bin/env python3
"""只做一次 PiPER MuJoCo observation → π0.5 inference 验证，不执行动作。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.environment import MujocoQualificationEnv, load_configs
from fragilegrab.benchmark.pi05_client import Pi05WebsocketClient
from fragilegrab.benchmark.pi05_contract import Pi05ContractError, PiperPi05Contract
from fragilegrab.benchmark.pi05_runner import load_rollout_config
from fragilegrab.benchmark.runner import native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--benchmark-config-dir", type=Path, default=ROOT / "configs/benchmark")
    parser.add_argument(
        "--object-config",
        type=Path,
        help="可选场景/物体配置；用于在 project 与 LIBERO 场景间显式切换",
    )
    parser.add_argument("--rollout-config", type=Path, required=True)
    parser.add_argument("--policy-host", required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--connect-attempts", type=int, default=1)
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="允许 execution_permitted=false 的 smoke normalization；仍严格禁止动作执行",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="可选的小型 JSON 报告；拒绝覆盖")
    args = parser.parse_args()
    robot_assets = args.robot_assets

    docs = load_configs(args.benchmark_config_dir, object_config=args.object_config)
    rollout = load_rollout_config(args.rollout_config)
    seed = docs["benchmark"]["values"]["seed"] if args.seed is None else args.seed
    contract = PiperPi05Contract(
        rollout["norm_stats_id"], action_horizon=rollout["action_horizon"]
    )

    env = MujocoQualificationEnv(docs, robot_assets)
    try:
        observation = env.reset(seed)
        request = contract.encode_observation(observation, rollout["prompt"])
        with Pi05WebsocketClient(
            args.policy_host,
            args.policy_port,
            connect_attempts=args.connect_attempts,
        ) as policy:
            contract.validate_server_metadata(
                policy.metadata,
                require_execution=not args.transport_only,
            )
            if args.transport_only:
                if policy.metadata.get("normalization_purpose") != "transport_only":
                    raise RuntimeError("--transport-only 只接受 normalization_purpose=transport_only")
                if policy.metadata.get("execution_permitted") is not False:
                    raise RuntimeError("--transport-only 要求服务明确禁止动作执行")
            started = time.monotonic_ns()
            response = policy.infer(request)
            elapsed_ms = (time.monotonic_ns() - started) / 1e6
            proposal = contract.decode_action_chunk_proposal(response)

            initial_target = env.data.ctrl.copy()
            previous = initial_target.copy()
            prefix_deltas = []
            validation_error = None
            validated_count = 0
            try:
                decoded = contract.decode_action_chunk(response, env.model.actuator_ctrlrange)
                for target in decoded[: rollout["execution_horizon"]]:
                    prefix_deltas.append(target - previous)
                    contract.validate_execution_step(
                        previous,
                        target,
                        max_joint_delta_rad=rollout["max_joint_delta_rad"],
                        max_gripper_delta_m=rollout["max_gripper_delta_m"],
                    )
                    previous = target
                    validated_count += 1
            except Pi05ContractError as exc:
                validation_error = str(exc)

            report = {
                "schema_version": "fragilegrab.pi05_inference_check/0.1",
                "status": (
                    "passed_transport_only_without_action_execution"
                    if args.transport_only
                    else "passed_without_action_execution"
                    if validation_error is None
                    else "rejected_without_action_execution"
                ),
                "transport_only": args.transport_only,
                "action_execution_attempted": False,
                "seed": seed,
                "policy_metadata": policy.metadata,
                "request": {
                    "base_image_shape": list(request["observation/image"].shape),
                    "base_image_dtype": str(request["observation/image"].dtype),
                    "base_image_sha256": hashlib.sha256(request["observation/image"].tobytes()).hexdigest(),
                    "wrist_image_shape": list(request["observation/wrist_image"].shape),
                    "wrist_image_dtype": str(request["observation/wrist_image"].dtype),
                    "wrist_image_sha256": hashlib.sha256(
                        request["observation/wrist_image"].tobytes()
                    ).hexdigest(),
                    "state": request["observation/state"],
                    "prompt": request["prompt"],
                },
                "response": {
                    "round_trip_ms": elapsed_ms,
                    "action_shape": list(np.asarray(response["actions"]).shape),
                    "initial_actuator_target": initial_target,
                    "decoded_first_proposal": proposal[0],
                    "decoded_nonexecuted_prefix": proposal[: rollout["execution_horizon"]],
                    "prefix_step_deltas": prefix_deltas,
                    "requested_validation_prefix": rollout["execution_horizon"],
                    "validated_nonexecuted_prefix": validated_count,
                    "validation_error": validation_error,
                },
            }
    finally:
        env.close()

    rendered = json.dumps(native(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"拒绝覆盖：{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image_dir = args.output.parent / f"{args.output.stem}_policy_images"
        if image_dir.exists():
            raise FileExistsError(f"拒绝覆盖：{image_dir}")
        image_dir.mkdir()
        cv2.imwrite(
            str(image_dir / "base_rgb.png"),
            cv2.cvtColor(request["observation/image"], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(image_dir / "wrist_rgb.png"),
            cv2.cvtColor(request["observation/wrist_image"], cv2.COLOR_RGB2BGR),
        )
        report["request"]["saved_policy_images"] = {
            "base_rgb": str(image_dir / "base_rgb.png"),
            "wrist_rgb": str(image_dir / "wrist_rgb.png"),
        }
        rendered = json.dumps(native(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] not in {
        "passed_without_action_execution",
        "passed_transport_only_without_action_execution",
    }:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
