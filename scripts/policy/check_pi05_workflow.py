#!/usr/bin/env python3
"""无需机器人资产，验证本地双 RGB/状态到云端 π0.5 action chunk 的数据流。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fragilegrab.benchmark.pi05_client import Pi05WebsocketClient  # noqa: E402
from fragilegrab.benchmark.pi05_contract import PiperPi05Contract  # noqa: E402


DEFAULT_NORM_STATS_ID = "piper_x_workflow_smoke_v0.1"


def _image_pair(height: int = 240, width: int = 320) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    base = np.stack(
        [np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width)), np.full((height, width), 48, np.uint8)],
        axis=-1,
    )
    wrist = np.stack(
        [np.full((height, width), 32, np.uint8), np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width))],
        axis=-1,
    )
    return base, wrist


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--norm-stats-id", default=DEFAULT_NORM_STATS_ID)
    parser.add_argument("--prompt", default="pick up the object carefully")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats 必须为正整数")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{args.output}")

    contract = PiperPi05Contract(args.norm_stats_id)
    base, wrist = _image_pair()
    request = contract.encode_observation(
        {
            "images": {"base_rgb": base, "wrist_rgb": wrist},
            "joint_position": np.array([0.0, 0.2, -0.3, 0.0, 0.1, 0.0, 0.05, -0.05]),
        },
        args.prompt,
    )
    latencies_ms: list[float] = []
    action_checksums: list[str] = []
    action_ranges: list[dict[str, float]] = []
    with Pi05WebsocketClient(args.policy_host, args.policy_port) as policy:
        contract.validate_server_metadata(policy.metadata, require_execution=False)
        if policy.metadata.get("normalization_purpose") != "transport_only":
            raise RuntimeError("该检查器只接受 normalization_purpose=transport_only")
        if policy.metadata.get("execution_permitted") is not False:
            raise RuntimeError("该检查器要求服务明确禁止动作执行")
        for _ in range(args.repeats):
            started = time.perf_counter()
            response = policy.infer(request)
            latency_ms = (time.perf_counter() - started) * 1000.0
            proposal = contract.decode_action_chunk_proposal(response)
            actions = np.asarray(response["actions"], dtype=np.float64)
            latencies_ms.append(latency_ms)
            action_checksums.append(_sha256_array(actions))
            action_ranges.append({"min": float(actions.min()), "max": float(actions.max())})
        metadata = policy.metadata

    result = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_without_action_execution",
        "action_execution_attempted": False,
        "server_metadata": metadata,
        "contract_id": metadata["contract_id"],
        "norm_stats_id": args.norm_stats_id,
        "normalization_purpose": metadata["normalization_purpose"],
        "request": {
            "base_rgb_shape": list(base.shape),
            "base_rgb_sha256": _sha256_array(base),
            "wrist_rgb_shape": list(wrist.shape),
            "wrist_rgb_sha256": _sha256_array(wrist),
            "state": request["observation/state"].tolist(),
            "prompt": request["prompt"],
        },
        "response": {
            "action_shape": list(proposal.shape),
            "action_sha256": action_checksums,
            "action_ranges": action_ranges,
        },
        "latency_ms": {
            "samples": latencies_ms,
            "mean": float(np.mean(latencies_ms)),
            "max": float(np.max(latencies_ms)),
        },
        "limitation": "只证明数据传输和 pi05_base 推理完成；输出未执行，也不证明 PiPER-X 抓取能力。",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
