#!/usr/bin/env python3
"""运行 Qualification 或 F1，或者执行两次同种子全链路 smoke test。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fragilegrab.benchmark.runner import run_episode, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-assets", type=Path, required=True, help="锁定 commit 的 agx_arm_sim checkout")
    p.add_argument("--config-dir", type=Path, default=ROOT / "configs/benchmark")
    p.add_argument("--object-config", type=Path, help="可选场景/物体配置")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=["qualification", "F1"], default="F1")
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true", help="重复同一 seed 并比较原始轨迹、RGB 与指标")
    args = p.parse_args()
    robot_assets = args.robot_assets
    if args.smoke:
        args.output.mkdir(parents=True, exist_ok=False)
        results = [run_episode(
            args.config_dir,
            robot_assets,
            args.output / str(i),
            args.task,
            args.seed,
            object_config=args.object_config,
        ) for i in range(2)]
        checks = {key: results[0]["integrity"][key] == results[1]["integrity"][key]
                  for key in ["trajectory_sha256", "raw_base_rgb_sha256",
                              "raw_wrist_rgb_sha256", "timeline_sha256"]}
        checks["metrics_equal"] = results[0]["result"] == results[1]["result"]
        for i in range(2):
            audit = json.loads((args.output / str(i) / "model_audit.json").read_text())
            for key in ["static_pass", "control_pass", "gripper_pass", "zero_in_range", "zero_finite",
                        "render_nonconstant", "wrist_render_nonconstant"]:
                checks[f"audit_{i}_{key}"] = audit[key]
        summary = {"pass": all(checks.values()), "checks": checks,
                   "task_success": results[0]["result"]["task_success"],
                   "说明": "smoke 验证工程全链路与重复性；task success 独立报告；不是 π0.5 结果"}
        save_json(args.output / "smoke_result.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["pass"]:
            raise SystemExit(1)
    else:
        result = run_episode(
            args.config_dir,
            robot_assets,
            args.output,
            args.task,
            args.seed,
            object_config=args.object_config,
        )
        print(json.dumps(result["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
