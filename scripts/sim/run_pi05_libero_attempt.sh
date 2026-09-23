#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="/tmp/fragilegrab_pi05_libero_seed7_attempt01"

if ! ss -ltn | grep -q '127.0.0.1:8000'; then
  echo "未检测到 127.0.0.1:8000；请先建立 π0.5 SSH 隧道。" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "输出目录已存在：${OUTPUT_DIR}" >&2
  echo "为避免覆盖证据，脚本不会继续。" >&2
  exit 3
fi

AVAILABLE_KB="$(df --output=avail /tmp | tail -n 1 | tr -d ' ')"
if (( AVAILABLE_KB < 102400 )); then
  echo "/tmp 可用空间不足 100 MB，停止运行。" >&2
  exit 4
fi

cd "${PROJECT_ROOT}"
env -u PYTHONPATH \
  NUMBA_CACHE_DIR=/tmp/fragilegrab-numba-cache \
  MUJOCO_GL=egl \
  .venv-mujoco/bin/python scripts/sim/run_pi05_mujoco.py \
    --robot-assets assets/vendor/agx_arm_sim \
    --object-config configs/benchmark/object.libero_spatial_bowl_to_plate.yaml \
    --rollout-config configs/pi05_mujoco_libero_spatial_zero_shot_diagnostic.yaml \
    --policy-host 127.0.0.1 \
    --policy-port 8000 \
    --seed 7 \
    --output "${OUTPUT_DIR}"

echo "运行完成：${OUTPUT_DIR}"
echo "全局相机视频：${OUTPUT_DIR}/base_rgb.mp4"
echo "腕部相机视频：${OUTPUT_DIR}/wrist_rgb.mp4"
