#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 ROBOT_ASSETS POLICY_HOST POLICY_PORT OUTPUT_DIR" >&2
  exit 2
fi
if [[ ${ALLOW_PI05_MUJOCO_EXECUTION:-} != "yes" ]]; then
  echo "refusing execution; set ALLOW_PI05_MUJOCO_EXECUTION=yes after inference-only checks pass" >&2
  exit 2
fi

robot_assets=$1
policy_host=$2
policy_port=$3
output_dir=$4
python_bin=${PYTHON_BIN:-.venv-mujoco/bin/python}
rollout_config=${ROLLOUT_CONFIG:-configs/pi05_mujoco_zero_shot_safe_test.yaml}

if [[ -e "$output_dir" ]]; then
  echo "output directory already exists: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"

for scene in project libero_spatial; do
  if [[ "$scene" == "project" ]]; then
    object_config="configs/benchmark/object.project.yaml"
    scene_rollout_config="$rollout_config"
  else
    object_config="configs/benchmark/object.libero_spatial_bowl_to_plate.yaml"
    scene_rollout_config="configs/pi05_mujoco_libero_spatial_zero_shot_diagnostic.yaml"
  fi
  env -u PYTHONPATH NUMBA_CACHE_DIR=.local_work/numba-cache MUJOCO_GL=egl \
    "$python_bin" scripts/sim/run_pi05_mujoco.py \
    --robot-assets "$robot_assets" \
    --object-config "$object_config" \
    --rollout-config "$scene_rollout_config" \
    --policy-host "$policy_host" \
    --policy-port "$policy_port" \
    --seed 7 \
    --output "$output_dir/${scene}_seed7"
done

echo "bounded zero-shot runs written to: $output_dir"
