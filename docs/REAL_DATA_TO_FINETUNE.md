# Piper X：从真机数采到 OpenPI 微调

本文是正式数据的操作清单。它建立在已验证通过的假数据微调链路之上，描述哪些内容已经可用、哪些内容必须在真实数采前补齐。

## 结论与边界

如果 D435i、DaBai DC1 与 Piper CAN 都能稳定工作，当前数采代码已经能直接写出符合 OpenPI Piper 配置的数据字段：双 RGB 图像、7 维 state、7 维 action、任务文本和对齐诊断信息。

但是，当前 `run_aligned_capture.py` 一次运行只创建一个新的 LeRobot 数据集，并在其中保存一个 episode；它拒绝复用已有 `--root`。因此：

- 可以立刻采集并验证单个真实 episode 的完整链路；
- 不能直接用同一命令把许多 episode 追加进同一个正式训练集；
- 在大规模正式数采前，需要实现“追加 episode 到同一数据集”，或实现经过验证的多目录 LeRobot episode 合并工具。

不要通过手工复制 Parquet/video 文件来合并 dataset；索引、episode metadata 和视频路径也必须一致。

## 0. 固定写入环境

本机数采必须使用：

```text
D:\vscode\opencv\venv\Scripts\python.exe
datasets==3.6.0
lerobot==0.1.0
```

云端训练使用同样的 `datasets==3.6.0`。此前由 `datasets 5.0.1` 写出的 LeRobot Parquet 元数据与云端读取端不兼容，不能用于训练。

## 1. 真机单 episode 数采

以下例子不发送任何机械臂控制命令；它只读取两台相机和 Piper CAN 状态：

```powershell
& D:\vscode\opencv\venv\Scripts\python.exe .\run_aligned_capture.py `
  --root D:\piper_x_real\data\real\pick_cube_v001_ep000 `
  --repo-id local/piper_x_pick_cube_v001 `
  --task "pick up the cube and place it in the container" `
  --seconds 30
```

可选地增加两台相机序列号：

```text
--d435i-serial <D435i序列号> --dc1-serial <DC1序列号>
```

运行逻辑：三个生产者每收到一条数据便附加电脑 `time.monotonic_ns()` 并放入时间窗口缓存；主循环从三路共同开始时刻后等待 100ms，在固定 30Hz 逻辑时刻寻找每路 ±30ms 内最近的样本。未满足窗口的逻辑时刻被丢弃，不会写入错配帧。

## 2. 数采输出与动作语义

每个 episode 根目录包含 LeRobot 标准的 `meta/`、`data/`、`videos/`，以及项目诊断文件：

```text
piper_capture_manifest.json
piper_raw/episode_000000_alignment.jsonl
```

LeRobot 特征：

| 字段 | 内容 |
|---|---|
| `observation.images.base` | D435i 原始 RGB 图像 |
| `observation.images.wrist` | DaBai DC1 原始 RGB 图像 |
| `observation.state` | `float32[7] = [q1..q6 rad, gripper_closedness]` |
| `action` | 同维度的下一逻辑时刻从臂状态代理 |
| `task_index` | LeRobot 中任务文本的索引；OpenPI 加载时转换为 prompt |

夹爪闭合度为 `1 - clip(width_m / 0.1001, 0, 1)`：0 表示全开，1 表示全闭。

动作的前六维是绝对关节位置，OpenPI 在训练输入端会计算 `action_q - state_q`；夹爪始终保持绝对闭合度。最后一帧动作等于自身 state，因此关节 delta 为 0。动作是观测到的下一时刻状态代理，若机械臂尚未到达真实控制目标，会与真实下发目标存在偏差；该风险当前只记录、不修改。

## 3. 每个 episode 的验收

在继续采集前检查：

1. 程序正常退出，未报告 producer failure。
2. 帧数约为采集时长 × 30Hz；偶发缺帧是时间窗口策略的预期结果。
3. `piper_capture_manifest.json` 显示 30Hz、±30ms、host clock 与正确相机角色。
4. `piper_raw/episode_000000_alignment.jsonl` 中的 `base_skew_ns`、`wrist_skew_ns`、`arm_skew_ns` 都处于 ±30ms 窗口内。
5. 随机播放/查看两路视频，确认相机角色没有互换、RGB 颜色正常、画面没有系统性裁剪。
6. 检查 state 与 action 的单位和范围：六关节为弧度；夹爪在 [0, 1]。

首次真机数采应只做一个短 episode，并将其上传到云端跑一次统计量和 20-step 冒烟，以验证本机写入版本没有偏差。

## 4. 组织正式多 episode 数据

为每个任务版本固定一个 `repo_id`，例如：

```text
local/piper_x_pick_cube_v001
```

同一个训练集中的 episode 应使用一致的相机安装、分辨率、state/action 定义和夹爪映射。任务文本可以不同，但必须准确描述演示任务。

在补齐追加/合并工具前，可先保存每个 episode 到独立根目录，完成验收后再合并；不要把不同采集条件的数据无标记混在一起。

## 5. 上传到云端

训练端不传入 `root`，LeRobot 将使用：

```text
/root/autodl-tmp/.cache/huggingface/lerobot/<repo_id>/
```

因此正式数据若使用 `repo_id=local/piper_x_pick_cube_v001`，云端目录应为：

```text
/root/autodl-tmp/.cache/huggingface/lerobot/local/piper_x_pick_cube_v001/
```

目录内直接放入最终合并后的 `meta/`、`data/`、`videos/` 等内容。云端必须保留系统级 FFmpeg，供 TorchCodec 解码 AV1 视频。

## 6. 新建正式 OpenPI 配置

保留 `LeRobotPiperDataConfig`、`PiperInputs`、`PiperOutputs`、图像映射和关节/夹爪变换不变。复制 `pi05_piper_debug_finetune` 为正式配置并修改：

| 配置项 | 正式数据应改为 |
|---|---|
| `name` | 新的实验配置名，例如 `pi05_piper_pick_cube_v001_finetune` |
| `repo_id` | 正式 LeRobot repo ID |
| `num_train_steps` | 根据正式数据规模和验证结果确定 |
| `batch_size` | 4090 显存可承受的值；当前 8 已验证能运行 |
| `exp_name` | 本次训练的可追溯实验名 |
| `CheckpointWeightLoader` | `/root/autodl-tmp/pi05_checkpoints/openpi-assets/checkpoints/pi05_base/params` |

不要复用全零假数据的 `assets/pi05_piper_debug_finetune/.../norm_stats.json`。

## 7. 计算正式统计量并启动微调

在云端 OpenPI 根目录执行：

```bash
cd /root/autodl-tmp/openpi
.venv/bin/python scripts/compute_norm_stats.py --config-name <正式配置名>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py <正式配置名> --exp-name <实验名> --overwrite
```

统计量写入该配置独立的 `assets/<配置名>/<repo_id>/norm_stats.json`。训练 checkpoint 写入：

```text
checkpoints/<配置名>/<实验名>/<step>/
```

首次正式数据仍建议先以较短训练步数验证数据读取、loss、checkpoint 与推理输入输出；确认后再启动完整训练。

## 8. 推理阶段不变的契约

真机推理 harness 必须向 Piper policy 提供：

```text
observation/image        D435i 原始 RGB HWC uint8
observation/wrist_image  DaBai DC1 原始 RGB HWC uint8
observation/state        float32[7]
prompt                   任务文本
```

不要在 harness 中重新裁剪或 resize 图像；policy 内已有等比例补边到 224×224 的变换。返回的前 7 维动作是 `[q1..q6, gripper_closedness]` 的绝对目标表示；控制安全、执行频率和 action chunk 触发策略将在真机推理阶段单独实现。
