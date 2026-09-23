#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 ROBOT_ASSETS POLICY_HOST POLICY_PORT OUTPUT_DIR" >&2
  exit 2
fi

robot_assets=$1
policy_host=$2
policy_port=$3
output_dir=$4
python_bin=${PYTHON_BIN:-.venv-mujoco/bin/python}
rollout_config=${ROLLOUT_CONFIG:-configs/pi05_mujoco_zero_shot.example.yaml}

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
  for seed in 7 17 27; do
    output="$output_dir/${scene}_seed${seed}.json"
    env -u PYTHONPATH NUMBA_CACHE_DIR=/tmp/fragilegrab-numba-cache MUJOCO_GL=egl \
      "$python_bin" scripts/sim/check_pi05_mujoco_inference.py \
      --robot-assets "$robot_assets" \
      --object-config "$object_config" \
      --rollout-config "$scene_rollout_config" \
      --policy-host "$policy_host" \
      --policy-port "$policy_port" \
      --seed "$seed" \
      --output "$output"
  done
done

echo "inference-only reports written to: $output_dir"
