# 真实 PiPER-X → 本地 MuJoCo 遥操作采集

## 范围与状态

该入口把真实 PiPER-X 当作**只读位置输入设备**。`pyAgxArm` 只执行 connect、joint/gripper feedback 和 disconnect；所有位置 target 只写入本地 MuJoCo。它不使能或控制真机，不做力反馈、双向控制、易碎阈值、安全控制或网络视频。

当前仓库验证了 mapping、限位、滤波、独立 reader、延迟统计，以及锁定 LIBERO-derived 场景的 MuJoCo 双相机/七维 action smoke。开发机已用 LeRobot 0.3.3 + CPU-only PyTorch 2.7.1 完成两帧双 RGB/Parquet/video/manifest 写盘 smoke；`pyAgxArm` 仍仅安装在系统 Python，且尚未发现 CAN 适配器或连接真机，因此“真机移动到保存一条 episode”和 50–100 ms 视觉反馈目标仍需现场验证。

## 数据与控制契约

控制链为：

```text
real q1..q6 + normalized gripper closedness
  → sign × scale + offset
  → configured joint/gripper clamp
  → optional first-order low-pass
  → MuJoCo actuator ctrlrange validation
  → q1..q6 rad + gripper actuator target m
  → MuJoCo step
```

LeRobot 的 `action` 是最后一行中**真正提交给 MuJoCo 的七维 target**，不是原始真机 qpos，也不是下一帧 sim qpos。`observation.state` 为 sim `[q1..q6 rad, gripper_closedness]`。front/wrist RGB、task prompt 随帧写入 LeRobot；host monotonic 的 read/apply/render 时间、leader state、sim state、mapping audit 和 action 同时写入 `teleop_timeline.jsonl`。

每条已保存 episode 的 `episode_manifest.json` 记录 scene seed、robot reset pose、object initial pose、success/failure、termination reason、有效 config hash、`pyAgxArm`/固件启动信息和 mean/P95 延迟。每条 episode 单独构成一个 LeRobot dataset，避免丢弃操作污染其他 episode。

## 配置

版本化模板是 `configs/teleop/piper_mujoco.yaml`。机器专属 CAN channel/interface 或现场标定值应复制到被忽略的 `configs/local/`，不要直接把机器配置或未经验证的标定提交为项目事实。

模板中的 mapping 是 Candidate：真实机械零位映射到当前 MuJoCo benchmark home。首次采集前必须在实际 PiPER-X 上逐关节小幅移动，确认方向、幅值和夹爪开闭，再校准：

- `mapping.joints.sign`
- `mapping.joints.scale`
- `mapping.joints.offset_rad`
- `mapping.joints.limits_rad`
- `mapping.gripper.*`
- `mapping.smoothing.enabled/alpha`

运行时还会拒绝任何超出实际 MuJoCo actuator `ctrlrange` 的配置。

## 环境与启动

在采集机的同一个 Python 环境中安装并核对现有 MuJoCo requirements、现场已验证版本的 `pyAgxArm` 和 LeRobot。当前本机验证组合是 LeRobot 0.3.3、`torch==2.7.1+cpu`、`torchvision==0.22.1+cpu`、NumPy 1.26.4 和 `opencv-python-headless==4.11.0.86`；CPU wheel 避免了默认 pip 解析下的大型 CUDA 依赖。不得凭本说明替换现场已验证的 CAN 驱动、Piper 固件或 SDK 组合。运行会把 `pyAgxArm` 包版本（若 package metadata 可用）和固件信息写入 manifest。

只需一个本地终端；reader、control/physics 和 render/record 在同一进程中用独立节拍运行：

```bash
MUJOCO_GL=glfw PYTHONPATH=src .venv-mujoco/bin/python \
  scripts/sim/run_piper_mujoco_teleop.py \
  --robot-assets assets/vendor/agx_arm_sim \
  --teleop-config configs/local/piper_mujoco.yaml \
  --object-config configs/benchmark/object.libero_spatial_bowl_to_plate.yaml \
  --output data/teleop_mujoco \
  --seed 7
```

`MUJOCO_GL=glfw` 需要本地图形桌面。不要用 `egl` 启动交互窗口。默认频率为 read 80 Hz、control 40 Hz、render/record 20 Hz；三者不会被锁为同一个同步频率。

MuJoCo 对每个相机只渲染一次 `640×480` RGB；该原图进入 LeRobot。操作窗口显示同一帧经 area downsample 得到的 `320×240` 预览，不触发第二次 MuJoCo render。LeRobot 使用 4 个异步 image writer thread，编码仍在 episode 结束时完成。训练/推理时由锁定 OpenPI transform 再把图像缩放到模型输入尺寸；存档不预先破坏为低分辨率。

## 热键

先点击 OpenCV 仿真窗口使其获得键盘焦点：

| 键 | 行为 |
| --- | --- |
| `S` | 开始一条 episode |
| `E` | 结束并保存为 unlabeled |
| `Y` | 结束、保存并标记 success |
| `F` | 结束、保存并标记 failure |
| `D` | 丢弃当前 episode |
| `R` | reset scene；录制中拒绝 reset |
| `Q` / `Esc` | 退出；活动中的未完成 episode 会丢弃 |

窗口左侧为第三人称 front，右侧为 wrist。当前版本同时显示两路 `320×240` 预览，没有单路切换模式；数据集中保存的是两路 `640×480` 原始渲染。

## 输出与检查

默认根目录为 `data/teleop_mujoco/`，已由 `.gitignore` 排除。成功保存后目录形如：

```text
episode_YYYYMMDD_HHMMSS_nnnnnnnnn/
  episode_manifest.json
  teleop_timeline.jsonl
  lerobot/
```

结束一条 episode 时终端打印 read→apply、read→render-finished 和 apply→render-finished 的 mean/P95。视觉反馈判断优先看 `read_to_render_finished`；它包括 leader sample age、控制等待、physics 和两路 rendering，但不包括显示器扫描与人的感知延迟。

现场最小验收：移动各真实关节和夹爪，确认 sim 同向/期望方向跟随；用 `S` 开始，完成一次操作后按 `Y` 或 `F`；检查 manifest 结果和时间字段，并随机核对 timeline 中 `action` 等于同帧 mapping audit 的 `executed_action`。
