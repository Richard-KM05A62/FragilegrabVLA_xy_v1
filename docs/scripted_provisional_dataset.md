# PiPER-X scripted provisional dataset

## 用途与边界

`scripts/sim/generate_scripted_provisional_dataset.py` 用于生成小规模 MuJoCo 数据，验证
PiPER-X/OpenPI 数据链路。生成器必须显式指定 `--config`，不提供默认采集计划。旧的
单场景 norm20 配置和专用统计/运行入口已经删除；历史数据只保留在外部存储中作为实验证据。

完整 8 模板/100 条任务计划见 `docs/piper_embodiment_task_plan.md`。旧
`gentle_demo_pilot.yaml` 仅保留为 pipeline smoke，不进入本轮 norm。

## 数据契约与检查

LeRobot 数据以 20 Hz 记录双 RGB、`[q1..q6 rad, gripper closedness]` state、实际提交给
MuJoCo 的 absolute `[q1..q6 rad, gripper target closedness]` action 和每条 episode 的 task prompt。
因果顺序是 `obs_t -> action_t -> physics -> obs_t+1`。

只有空 episode、NaN/Inf、shape 错误、时间戳不一致、image/state/action/video 长度不一致、
超出模型硬关节限位、写盘或序列化失败可以阻断 LeRobot 提交。其他表现量只记为
metric/warning，不删除原始 episode；metric threshold 默认为空且关闭。

## 执行状态

新的 20 条计划需先接入 5 个官方 LIBERO 场景，见
`docs/piper_embodiment_task_plan.md`。以下仅保留生成器的通用调用格式：

列出计划，不生成数据：

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python \
  scripts/sim/generate_scripted_provisional_dataset.py \
  --config /path/to/an/enabled_plan.yaml --list-variations
```

启用新计划后，预检和完整 run 都必须显式传入该版本化 config；不允许无 config 运行。

如需复现历史 smoke，必须显式加
`--config configs/benchmark/gentle_demo_pilot.yaml`。输出目录不得预先存在；其中包含
`lerobot/`、effective config 副本、summary 和每条 episode 的 raw audit 目录。

## 计算 normalization

不再使用项目侧 raw timeline 统计脚本生成可执行 normalization。正式统计只能由锁定 OpenPI
commit 的官方 `compute_norm_stats.py` 完成，并且必须通过 Piper-specific DataConfig、transform 和
LeRobot loader 实读最终 scripted + teleop train split。统计产物必须绑定 dataset version、transform id、
OpenPI commit、action horizon 和后续微调 checkpoint。
