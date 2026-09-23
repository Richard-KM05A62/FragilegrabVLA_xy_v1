# PiPER π0.5 接入手册

当前部署锁定 AutoDL 已验证的 OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479` 和官方 `pi05_base` checkpoint。该 commit 是本地接口分析基线 `15a9616a00943ada6c20a0f158e3adb39df2ccac` 的直接子提交，上游只修改了 DROID 文档拼写，相关接口代码一致。契约 `piper_x_joint_position_dual_rgb_v0.3` 使用全局 RGB + 夹爪中部 D435 RGB，有效接口是七维：六轴关节角（rad）加夹爪闭合度 `[0,1]`；模型内部仍保留 32 维 padding 和 50-step action chunk。Normalization 必须使用 `piper_x_` 前缀并携带匹配的 provenance。

## 0. 最短 workflow smoke（已完成，仅供重现）

这条路径不依赖机器人或 MuJoCo 资产，只验证：本地双 RGB + 七维 state + prompt 能否送达云端，云端能否用官方 `pi05_base` 返回 `(50, 7)` action chunk。合成 normalization 带有 `execution_permitted=false`；所有闭环 runner 会拒绝它，结果不得用于动作执行或表述为 PiPER-X 抓取能力。

本地生成临时统计：

```bash
.venv-mujoco/bin/python scripts/policy/create_pi05_workflow_smoke_stats.py \
  --output assets/local/piper_x_workflow_smoke_v0.1
```

把 `scripts/policy/serve_pi05_piper.py` 和整个 `assets/local/piper_x_workflow_smoke_v0.1/` 目录传到云端，然后在锁定 OpenPI 环境启动：

```bash
.venv/bin/python /path/to/serve_pi05_piper.py \
  --checkpoint-dir /root/autodl-tmp/cache/openpi/openpi-assets/checkpoints/pi05_base \
  --norm-stats-dir /path/to/piper_x_workflow_smoke_v0.1 \
  --norm-stats-id piper_x_workflow_smoke_v0.1 \
  --host 127.0.0.1 \
  --port 8000
```

若云端通过 SSH 访问，在本地另开终端建立端口转发（替换登录地址）：

```bash
ssh -N -L 8000:127.0.0.1:8000 user@cloud-host
```

最后在本地运行资产无关的检查器：

```bash
.venv-mujoco/bin/python scripts/policy/check_pi05_workflow.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --output outputs/pi05_workflow_smoke.json
```

成功标志是输出中的 `status` 为 `passed_without_action_execution`、`action_shape` 为 `[50, 7]`，并记录至少一条推理延迟。这里的 `execution_permitted` 只是防误用的软件门禁，不是 Safety Layer。

## 1. 生成本机校准统计

```bash
python3 scripts/policy/compute_piper_norm_stats.py \
  /path/to/run_a /path/to/run_b /path/to/run_c \
  --output assets/local/piper_x_mujoco_scripted_calibration_v0.1 \
  --norm-stats-id piper_x_mujoco_scripted_calibration_v0.1
```

该产物只做输入/输出尺度校准，不修改权重，也不是 Piper 真机数据分布的证明。`assets/local/` 不进入 Git。

## 2. 在 OpenPI 机器启动专用服务

从已锁定 commit 的 OpenPI 虚拟环境中运行：

```bash
.venv/bin/python /path/to/FragilegrabVLA_xy_v1/scripts/policy/serve_pi05_piper.py \
  --checkpoint-dir /root/autodl-tmp/cache/openpi/openpi-assets/checkpoints/pi05_base \
  --norm-stats-dir /path/to/piper_x_mujoco_scripted_calibration_v0.1 \
  --norm-stats-id piper_x_mujoco_scripted_calibration_v0.1 \
  --host 127.0.0.1 \
  --port 8000
```

服务启动时会核对 OpenPI commit。如果推理机不在本机，建议用 SSH tunnel 把远程 `127.0.0.1:8000` 映射到本机，不要直接把无认证端口暴露到公网。

## 3. 先做不执行动作的推理检查

```bash
env -u PYTHONPATH NUMBA_CACHE_DIR=/tmp/fragilegrab-numba-cache MUJOCO_GL=egl \
  .venv-mujoco/bin/python scripts/sim/check_pi05_mujoco_inference.py \
  --robot-assets /path/to/agx_arm_sim \
  --rollout-config configs/pi05_mujoco_zero_shot.example.yaml \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --output outputs/pi05_inference_check.json
```

这一步只验证 metadata、observation、`(50,7)` 输出、反归一化以及配置指定的非执行前缀校验，不向 MuJoCo 提交任何动作。

## 4. 运行 MuJoCo 最小闭环

只有第 3 步通过后再运行：

```bash
env -u PYTHONPATH NUMBA_CACHE_DIR=/tmp/fragilegrab-numba-cache MUJOCO_GL=egl \
  .venv-mujoco/bin/python scripts/sim/run_pi05_mujoco.py \
  --robot-assets /path/to/agx_arm_sim \
  --rollout-config configs/pi05_mujoco_zero_shot.example.yaml \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --output runs/pi05_f1_seed7
```

严格示例配置每次保存完整 50-step chunk，执行多少步由对应 rollout config 的 `execution_horizon` 决定，不应将“两步”视为系统常量。越关节范围或越单步变化阈值时严格模式直接拒绝，不做静默裁剪。

## 5. 运行 zero-shot 投影诊断闭环

严格模式证明原始 `pi05_base` 输出会被拒绝后，可使用独立配置观察它在 MuJoCo 中的受限行为：

```bash
env -u PYTHONPATH NUMBA_CACHE_DIR=/tmp/fragilegrab-numba-cache MUJOCO_GL=glfw \
  .venv-mujoco/bin/python scripts/sim/run_pi05_mujoco.py \
  --robot-assets /path/to/agx_arm_sim \
  --rollout-config configs/pi05_mujoco_zero_shot_projected.example.yaml \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --output runs/pi05_zero_shot_projected_seed7 \
  --seed 7 \
  --display
```

该配置每次最多执行 chunk 前 5 步，将关节范围和单步变化投影分别记录，并在非预期碰撞或诊断接触力上限触发时停止。它只用于未微调模型的 MuJoCo 能力边界实验；投影后的动作不得表述为模型原生输出，也不得用于真机。
