#!/usr/bin/env python3
"""运行 pi05_base → PiPER MuJoCo 的最小闭环 episode。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.pi05_client import Pi05WebsocketClient
from fragilegrab.benchmark.pi05_runner import run_pi05_episode


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-host", required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--api-key-env", help="保存 API key 的环境变量名；不在命令行传 key 本身")
    parser.add_argument("--connect-attempts", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--display", action="store_true", help="实时显示全局与腕部策略输入；Q/Esc 停止")
    args = parser.parse_args()
    robot_assets = args.robot_assets
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and api_key is None:
        raise RuntimeError(f"环境变量未设置：{args.api_key_env}")

    with Pi05WebsocketClient(
        args.policy_host,
        args.policy_port,
        api_key=api_key,
        connect_attempts=args.connect_attempts,
    ) as policy:
        manifest = run_pi05_episode(
            args.benchmark_config_dir,
            args.rollout_config,
            robot_assets,
            args.output,
            policy,
            seed=args.seed,
            display=args.display,
            object_config=args.object_config,
        )
    print(json.dumps(manifest["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
