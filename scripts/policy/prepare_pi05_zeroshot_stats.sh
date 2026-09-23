#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ROBOT_ASSETS OUTPUT_ROOT" >&2
  exit 2
fi

robot_assets=$1
output_root=$2
python_bin=${PYTHON_BIN:-.venv-mujoco/bin/python}
norm_stats_id=${NORM_STATS_ID:-piper_x_mujoco_scripted_calibration_v0.1}
episode_root="$output_root/episodes"
stats_root="$output_root/$norm_stats_id"

if [[ -e "$output_root" ]]; then
  echo "output root already exists: $output_root" >&2
  exit 2
fi
mkdir -p "$episode_root"

for scene in project libero_spatial; do
  if [[ "$scene" == "project" ]]; then
    object_config="configs/benchmark/object.project.yaml"
  else
    object_config="configs/benchmark/object.libero_spatial_bowl_to_plate.yaml"
  fi
  for seed in 7 17 27; do
    env -u PYTHONPATH NUMBA_CACHE_DIR=/tmp/fragilegrab-numba-cache MUJOCO_GL=egl \
      "$python_bin" scripts/sim/run_mujoco_benchmark.py \
      --robot-assets "$robot_assets" \
      --object-config "$object_config" \
      --task qualification \
      --seed "$seed" \
      --output "$episode_root/${scene}_seed${seed}"
  done
done

"$python_bin" scripts/policy/compute_piper_norm_stats.py \
  "$episode_root" \
  --output "$stats_root" \
  --norm-stats-id "$norm_stats_id"

echo "temporary zero-shot calibration stats written to: $stats_root"
