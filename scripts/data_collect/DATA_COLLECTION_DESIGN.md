# Piper X 数采链路：当前实现与数据规范

本文记录目前已实现的 Piper X 双相机数采链路，以及数据进入 OpenPI 微调前后的约定。

## 1. 当前状态

已完成：

- Piper X 的只读 CAN 状态读取与夹爪反馈读取；
- D435i 与 DaBai DC1 的相机适配器；
- 基于电脑单调时钟的三路缓存与 30 Hz 近邻对齐；
- 对齐样本直接写入 LeRobot 数据集；
- 单命令启动入口和采集线程错误回传。

尚未完成实际双相机出图与整条链路的现场验证；代码不包含任何机械臂使能、运动、夹爪控制、复位或标定调用。

## 2. 代码结构

| 文件 | 作用 |
| --- | --- |
| [aligned_capture.py](./aligned_capture.py) | 相机/CAN 读取器、时间缓存、对齐主循环、LeRobot writer。 |
| [run_aligned_capture.py](./run_aligned_capture.py) | 命令行启动入口。 |
| [collect_piper_lerobot.py](./collect_piper_lerobot.py) | 早期单相机直接采集脚本；不使用双相机同步主循环。 |

正式采集使用 `run_aligned_capture.py`。

## 3. 数据源读取

### 3.1 Piper X CAN 与夹爪

`PiperCanAdapter` 通过当前确认可用的 SDK 配置创建只读连接：

```python
create_agx_arm_config(
    robot="piper_x",
    comm="can",
    channel="0",
    interface="agx_cando",
)
```

读取流程：

1. `robot.connect()` 建立 CAN 通信；
2. `robot.get_joint_angles()` 返回六个从臂关节反馈，单位为 rad；
3. `robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)` 注册夹爪反馈解析器；
4. `effector.get_gripper_status()` 读取夹爪宽度（m）、力（N）与状态；
5. 每次读取到最新缓存状态后，用 `time.monotonic_ns()` 写入软件时间戳。

SDK 的 `MessageAbstract.timestamp` 仍记录到原始诊断日志中，但**不参与首版同步**。

夹爪模型值为：

```text
g_closed = 1 - clip(width_m / 0.1001, 0, 1)

0.0 = 全开
1.0 = 全闭
```

`0.1001 m` 来自本机只读检查到的当前全开附近反馈；在后续受控开闭标定后可替换为最终量程。

### 3.2 Intel RealSense D435i：外部视角

`RealSenseD435iAdapter` 使用 `pyrealsense2`：

- 可选按序列号绑定；
- 优先请求 `640×480 / RGB8 / 30 FPS`；
- 若 RGB8 profile 不可用，回退到 BGR8 并在适配器内转换为 RGB；
- `wait_for_frames()` 返回后立即记录 `time.monotonic_ns()`；
- 复制 SDK frame buffer，输出独立的 `uint8[H,W,3]` RGB 数组。

当前虚拟环境已安装 `pyrealsense2 2.58.4.10922`。

### 3.3 Orbbec DaBai DC1：腕部视角

`OrbbecDaBaiDC1Adapter` 通过 `ctypes` 调用已安装的 Orbbec C/C++ SDK：

```text
C:/Users/13302/Downloads/
OrbbecSDK_C_C++_v1.10.37_20260707_3f75820b8_win_x64_release/
OrbbecSDK_v1.10.37/SDK/lib/OrbbecSDK.dll
```

读取流程：

1. 创建 Pipeline；若提供序列号则按序列号选取设备；
2. 优先配置 `640×480 / RGB / 30 FPS`，无匹配时寻找任意 RGB profile；
3. 启动 Pipeline；
4. `ob_pipeline_wait_for_frameset()` 返回后立即记录 `time.monotonic_ns()`；
5. 取得 color frame，复制 SDK buffer 成 `uint8[H,W,3]` RGB 数组；
6. 释放 SDK frame，关闭时停止并释放 Pipeline、profile、device 和 context。

## 4. 时间同步主循环

### 4.1 时间域与缓存

首版只采用电脑单调时钟 `time.monotonic_ns()`。三个生产线程独立运行：

```text
D435i RGB 线程 ─┐
DC1 RGB 线程   ─┼─> 各自的 2 秒时间窗口缓存
Piper CAN 线程 ─┘
```

每条新数据进入缓存时都保存：

```text
(data, host_monotonic_timestamp_ns)
```

缓存按时间窗口裁剪：删除早于当前数据时间 `2 s` 的条目，而不是使用固定帧数。

### 4.2 首帧与固定采样时钟

当三路各自至少收到一条数据后：

```text
common_start = max(base_first_ts, wrist_first_ts, can_first_ts)
T0 = common_start + 100 ms
```

程序等待到 `T0 + 30 ms`，保证缓存已覆盖 `T0` 两侧，再在每一路中选离 `T0` 最近的数据。

后续目标时刻固定为：

```text
T_k = T0 + k / 30 Hz
```

主循环在电脑时间达到 `T_k + 30 ms` 后处理本 tick。这个 30 ms 输出延迟允许近邻选择来自目标时刻前后两侧。

### 4.3 选择规则

对每个 `T_k`：

```text
base  = nearest(base_buffer, T_k)
wrist = nearest(wrist_buffer, T_k)
arm   = nearest(arm_buffer, T_k)
```

仅在三路都满足下式时形成样本：

```text
abs(source_timestamp - T_k) <= 30 ms
```

若任一路不满足，跳过该 tick，不复制旧图填补。生产线程的异常会传回启动脚本，而不是无限等待 `T0`。

## 5. LeRobot 输出格式

每个 episode 写为一个 LeRobot 数据集。默认图像 feature 类型为 `video`；使用 `--images` 可改为逐图像保存。

| 字段 | 类型 | 内容 |
| --- | --- | --- |
| `observation.images.base` | RGB `uint8[H,W,3]` | D435i 外部视角，实际原始分辨率。 |
| `observation.images.wrist` | RGB `uint8[H,W,3]` | DaBai DC1 腕部视角，实际原始分辨率。 |
| `observation.state` | `float32[7]` | `[q1, q2, q3, q4, q5, q6, g_closed]`。关节单位 rad。 |
| `action` | `float32[7]` | 下一条对齐从臂状态，见下文。 |
| `task` | string | 本 episode 的自然语言任务描述。 |

采集阶段不裁剪、不缩放、不归一化图像。LeRobot 内部时间轴由帧号生成：

```text
timestamp = frame_index / 30
```

真实逻辑时刻不写入训练 feature，而记录在 sidecar 中。

### 5.1 动作标签

主臂命令直接发往从臂，不经过电脑；电脑无法读到真实控制命令。因此当前 action 是显式标记的代理标签：

```text
action[t] = aligned_follower_state[t + 1]
```

即以后一条对齐从臂反馈作为本帧的绝对动作目标。最后一帧采用：

```text
action[last] = state[last]
```

数据集 manifest 中会记录 `action_source = next_observation_proxy`。以后若接入可读取的真实命令流，只需替换 action 生成器，不需要改变 image/state schema。

#### 已知风险：动作代理与实际控制目标不完全等价

当前 `action[t] = follower_state[t + 1]` 是可观测的运动结果代理，而不是主臂真正发送给从臂的控制目标。若真实控制指令在下一采样时刻尚未达到、从臂仍在运动、或存在主从通信与机械响应延迟，则：

```text
follower_state[t + 1] != commanded_target[t]
```

这会使当前 action 标签更接近“一个采样周期后的实际位置”，而非原始控制意图，并可能低估快速动作的目标增量。该风险在本阶段只做记录，不修改采集策略；日后若能够读取主臂命令或从臂目标指令，应以该真实命令流替换代理 action。

### 5.2 Sidecar

数据集根目录额外输出：

```text
piper_capture_manifest.json
piper_raw/episode_000000_alignment.jsonl
```

其中包括：

- 夹爪映射、相机角色、时间同步与数据版本说明；
- 每帧逻辑目标时间；
- base、wrist、CAN 相对目标时刻的偏差；
- 原始夹爪宽度、夹爪力、SDK 时间戳等诊断信息。

当前运行脚本尚未把固件字符串写入 manifest；Piper 固件已在独立只读诊断中确认，后续可作为 manifest 的追加字段。

## 6. 启动方式

在指定虚拟环境中执行：

```powershell
& 'D:\vscode\opencv\venv\Scripts\python.exe' D:\piper_x_real\run_aligned_capture.py `
  --root D:\piper_x_real\data\episode_001 `
  --repo-id local/piper_x_teleop `
  --task "将红色物块放入容器" `
  --seconds 30
```

可选参数：

```text
--d435i-serial <D435i 序列号>
--dc1-serial <DC1 序列号>
--images                         # 不编码视频，保存图像 feature
```

`--root` 必须是尚不存在的新目录。

## 7. OpenPI 微调约定

采集数据保留绝对关节状态与绝对代理动作。进入 OpenPI 的 Piper 数据适配层后：

```text
关节 action 前六维：q_target - q_current
夹爪 action 第七维：仍为绝对 g_closed
```

随后使用训练集统计量进行分位数归一化；模型内部会将 7 维 Piper state/action 填充至 π0.5 的 32 维输入输出。图像在 OpenPI 模型端保持比例缩放并 padding 到 `224×224`；推理阶段只保留确定性视觉预处理。

## 8. 已验证与待验证项

| 项目 | 状态 |
| --- | --- |
| Piper CAN 连接、关节/夹爪只读反馈 | 已在真机验证。 |
| Piper 固件读取 | 已在真机验证：软件版本 `S-V1.9-0`。 |
| DC1 Orbbec SDK DLL 加载 | 已离线验证。 |
| D435i `pyrealsense2` 导入 | 已离线验证。 |
| LeRobot 双图像 + 7D state/action 写盘 | 已离线 smoke test。 |
| D435i 真机出 RGB 图 | 待相机到位后验证。 |
| DC1 真机出 RGB 图 | 待相机到位后验证。 |
| 三路对齐与完整 episode 真机录制 | 待相机到位后验证。 |
| 夹爪最终全开/全闭量程标定 | 待受控测试。 |
| OpenPI Piper transform 与云端微调配置 | 待在云端 OpenPI 环境接入。 |
