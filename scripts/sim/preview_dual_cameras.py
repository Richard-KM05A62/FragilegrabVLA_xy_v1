#!/usr/bin/env python3
"""实时显示将发给 π0.5 的 MuJoCo 全局+腕部 RGB；不连接 policy server。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.environment import MujocoQualificationEnv, load_configs
from fragilegrab.benchmark.runner import script_plan


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(bgr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/benchmark")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--motion", choices=("stationary", "scripted"), default="scripted")
    args = parser.parse_args()
    robot_assets = args.robot_assets

    docs = load_configs(args.config_dir)
    env = MujocoQualificationEnv(docs, robot_assets)
    paused = False
    try:
        observation = env.reset(args.seed)
        plan = script_plan(env, "F1") if args.motion == "scripted" else []
        phase_index = 0
        phase_step = 0
        phase_start = env.data.ctrl.copy()
        dt = env.model.opt.timestep
        control_steps = env.cfg["control_steps"]
        window = "FragileGrabVLA policy inputs | SPACE pause | R reset | Q quit"

        while True:
            phase = "stationary"
            if not paused and plan:
                phase, target, duration = plan[phase_index]
                phase_total = max(1, round(duration / (dt * control_steps)))
                fraction = min(1.0, (phase_step + 1) / phase_total)
                action = phase_start + fraction * (target - phase_start)
                for _ in range(control_steps):
                    env.step_physics(action)
                phase_step += 1
                if phase_step >= phase_total:
                    phase_index = (phase_index + 1) % len(plan)
                    phase_step = 0
                    phase_start = action.copy()
                    if phase_index == 0:
                        observation = env.reset(args.seed)
                        phase_start = env.data.ctrl.copy()
                observation = env.observe()

            base = _label(observation["images"]["base_rgb"], f"base_0_rgb | {phase}")
            wrist = _label(observation["images"]["wrist_rgb"], f"left_wrist_0_rgb | {phase}")
            cv2.imshow(window, np.concatenate([base, wrist], axis=1))
            key = cv2.waitKey(max(1, round(1000 * dt * control_steps))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                observation = env.reset(args.seed)
                phase_index = phase_step = 0
                phase_start = env.data.ctrl.copy()
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
