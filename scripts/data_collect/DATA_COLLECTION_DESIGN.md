# Piper X 数采链路：实现说明

本文描述 `aligned_capture.py` 与 `run_aligned_capture.py` 当前代码的行为，供审核和现场验证使用。这里列出的数值是实现快照，不代表经过实验确认的项目标准。跨模块数据约定见 [`../../docs/data_format.md`](../../docs/data_format.md)，验证状态见 [`../../docs/status.md`](../../docs/status.md)。

## 1. 实现边界

当前代码包含：

- Piper X 关节和夹爪状态的只读 CAN 读取；
- Intel RealSense D435i RGB 读取；
- Orbbec DaBai DC1 RGB 读取；
- 三路主机单调时钟缓存和近邻对齐；
- 单个 episode 的 LeRobot 数据集及 sidecar 写盘；
- 采集线程异常回传。

当前代码不包含机械臂使能、运动、夹爪控制、复位或标定。D435i depth 尚未采集。仓库中不存在旧说明曾引用的 `collect_piper_lerobot.py`。

## 2. 代码入口

| 文件 | 作用 |
| --- | --- |
| `aligned_capture.py` | 相机和 CAN 读取器、时间缓存、对齐循环与 LeRobot writer。 |
| `run_aligned_capture.py` | 一次只读采集的命令行入口。 |

`run_aligned_capture.py` 要求新的输出目录、task 文本和采集时长，并允许传入两个相机序列号及图像存储方式。当前每次运行创建一个新的 LeRobot 数据集，并保存一个 episode。

## 3. 数据源

### 3.1 Piper X

`PiperCanAdapter` 当前使用 `pyAgxArm` 创建 Piper X CAN 连接，并通过 `get_joint_angles()` 和夹爪 effector 读取反馈。每条状态包含：

```text
[joint_1_rad, joint_2_rad, joint_3_rad,
 joint_4_rad, joint_5_rad, joint_6_rad,
 gripper_closedness]
```

当前夹爪换算为：

```text
gripper_closedness = 1 - clip(width_m / open_width_m, 0, 1)
```

代码中的 `open_width_m` 默认值是现场标定前的实现值。固件与 `pyAgxArm` 配置必须按实际硬件版本复核；`start()` 取得的固件和状态当前没有写入 episode manifest。

### 3.2 D435i 外部相机

`RealSenseD435iAdapter` 通过 `pyrealsense2` 请求彩色流，优先 RGB8，在相应 profile 失败时尝试 BGR8 并转换为 RGB。代码复制 SDK frame buffer，输出独立的 HWC `uint8` 数组。

相机分辨率、帧率、SDK 版本和设备序列号目前没有完整写入 manifest。depth、内参、外参和相机时间戳尚未进入训练数据。

### 3.3 DaBai DC1 腕部相机

`OrbbecDaBaiDC1Adapter` 通过 `ctypes` 调用 Orbbec C SDK。类构造参数可以传入 SDK 根目录，但当前命令行入口没有暴露该参数，源码中仍有开发机默认路径；迁移配置前不得把它视为可移植设置。

代码请求 RGB profile 并复制 color frame buffer。资源释放路径需要依据实际使用的 Orbbec SDK 版本重新核对，特别是从 frameset 取得的 color frame 所有权。

## 4. 当前时间对齐

三个读取线程在数据到达应用侧时使用 `time.monotonic_ns()`。每路数据保存在按时间窗口裁剪的缓存中。三路均取得首样本后，调度器建立公共起点，并按固定逻辑 tick 选择每个缓存中最接近目标时刻的样本；任一路偏差超限时跳过该 tick。

当前源码常量包括 nominal FPS、最大允许偏差、启动延迟和缓存窗口。它们需要通过真实数据的偏差分布和动作时序验证，再迁移到配置。

LeRobot 的帧时间由连续 `frame_index / fps` 构造。由于未写入的逻辑 tick 会被压缩，训练时间轴可能与实际经过时间不同；真实目标时刻和三路偏差保存在 sidecar 中。

## 5. 当前 LeRobot 输出

| 字段 | 类型和语义 |
| --- | --- |
| `observation.images.base` | D435i 外部视角 RGB，实际输出尺寸。 |
| `observation.images.wrist` | DaBai DC1 腕部视角 RGB，实际输出尺寸。 |
| `observation.state` | 六个关节角（rad）和夹爪闭合量。 |
| `action` | 下一条对齐从臂状态代理。 |
| `task` | episode 的自然语言任务。 |

数据集根目录还写入 `piper_capture_manifest.json` 和 `piper_raw/episode_000000_alignment.jsonl`，用于保存 schema、当前采集参数、时间偏差和原始夹爪诊断等信息。manifest 尚未记录采集代码 commit 和完整依赖版本。

## 6. Action 语义与风险

当前电脑端没有读取到主臂实际控制命令，因此代码使用：

```text
action[t] = aligned_follower_state[t + 1]
action[last] = state[last]
```

这表示一个采样周期后的可观测从臂状态代理，不等于 `commanded_target[t]`。控制通信、机械响应和近邻对齐都会造成差异。manifest 必须保留 `action_source = next_observation_proxy`；在读取到真实命令流或确认新的动作定义前，不应删除这一限定。

## 7. 与 OpenPI 的关系

当前数据保留七维 Piper 状态和代理 action。Piper 专用 OpenPI repack、输入/输出 transform、归一化和 checkpoint 绑定尚未在本仓库实现。

任何转换都应以锁定的 OpenPI commit 和真实 episode 为依据。OpenPI 模型内部的 state/action padding 与 Piper 有效维度应使用对应版本的官方机制。是否将前六维 action 转为关节增量、夹爪保持绝对量，只能在实际控制语义确认后决定，当前为 TBD。

## 8. 验证状态

### 仓库静态检查已确认

- 两个 Python 文件可被语法解析；
- 调用链是只读采集；
- 当前双 RGB、七维 state/action 和 sidecar 写盘路径存在；
- 对齐使用主机单调时钟和近邻选择。

### 队友报告，待补仓库证据

- Piper CAN 和夹爪反馈可读取；
- Piper 固件版本已读取；
- D435i Python SDK 可导入；
- Orbbec SDK 动态库可加载；
- LeRobot 双图像和七维 state/action 做过离线 smoke test。

这些报告需要补充环境清单、命令、日志摘要或数据 manifest 后，才能标记为仓库 Verified。

### 尚待验证

- 两台相机同时稳定输出；
- 三路对齐与完整 episode 现场写盘；
- 夹爪全开/全闭量程；
- 不完整 episode 的处理；
- depth、相机标定与设备时间戳；
- Piper OpenPI transform 和 π0.5 数据加载。

## 9. 运行配置规则

实际 Python 环境、SDK 根目录、CAN 通道、设备序列号、输出目录、采集时间和采样参数必须由队友依据目标机器填写。可共享默认值应在现场验证后进入 `configs/`；个人值放入 `configs/local/`。本文不保存个人绝对路径或把尚未确认的命令作为官方启动方式。
