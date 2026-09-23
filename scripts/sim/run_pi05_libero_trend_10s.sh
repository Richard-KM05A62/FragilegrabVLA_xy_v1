#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="/tmp/fragilegrab_pi05_libero_seed7_trend_10s"

if ! ss -ltn | grep -q '127.0.0.1:8000'; then
  echo "未检测到 127.0.0.1:8000；请先建立 π0.5 SSH 隧道。" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  if [[ "${OUTPUT_DIR}" != /tmp/fragilegrab_pi05_libero_seed7_trend_10s ]]; then
    echo "拒绝删除非预期输出目录：${OUTPUT_DIR}" >&2
    exit 3
  fi
  echo "覆盖旧的趋势测试：${OUTPUT_DIR}"
  rm -rf -- "${OUTPUT_DIR}"
fi

AVAILABLE_KB="$(df --output=avail /tmp | tail -n 1 | tr -d ' ')"
if (( AVAILABLE_KB < 204800 )); then
  echo "/tmp 可用空间不足 200 MB，停止运行。" >&2
  exit 4
fi

cd "${PROJECT_ROOT}"
env -u PYTHONPATH \
  NUMBA_CACHE_DIR=/tmp/fragilegrab-numba-cache \
  MUJOCO_GL=egl \
  .venv-mujoco/bin/python scripts/sim/run_pi05_mujoco.py \
    --robot-assets assets/vendor/agx_arm_sim \
    --object-config configs/benchmark/object.libero_spatial_bowl_to_plate.yaml \
    --rollout-config configs/pi05_mujoco_libero_spatial_zero_shot_trend_10s.yaml \
    --policy-host 127.0.0.1 \
    --policy-port 8000 \
    --seed 7 \
    --output "${OUTPUT_DIR}"

echo "运行完成：${OUTPUT_DIR}"
echo "全局 MP4：${OUTPUT_DIR}/base_rgb.mp4"
echo "腕部 MP4：${OUTPUT_DIR}/wrist_rgb.mp4"
