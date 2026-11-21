# Linux H40 RTC 推理 Runbook

本文给出当前 Linux H40 推理系统的标准运行流程。环境安装见 [Linux H40 RTC 推理迁移手册](../environment/LINUX_INFERENCE_MIGRATION_GUIDE.md)。

## 1. 运行结构与脚本

系统包含两个本机进程：模型服务使用 GPU 生成动作，设备客户端采集相机和 CAN 状态、执行 RTC 调度，并在显式启用真机模式时控制机械臂。

| 文件 | 作用 |
|---|---|
| `inference/serve_piper_rtc.py` | 启动 JAX/GPU WebSocket 模型服务 |
| `inference/piper_rtc_policy.py` | 加载 OpenPI checkpoint，执行 pi0.5 RTC 推理 |
| `inference/run_piper_rtc_linux.py` | Linux RTC 主入口，选择只读或真机执行器 |
| `inference/run_piper_rtc.py` | 通用运行循环和命令行参数 |
| `inference/rtc_controller.py` | Bootstrap、RTC 请求和滚动计划 |
| `inference/rtc_core.py` | 时钟、数据缓存、action 队列和 Executor |
| `inference/rtc_transport.py` | WebSocket 通信及 horizon/stats 契约校验 |
| `inference/rtc_sources_linux.py` | 把 Linux 相机和 CAN 接入 RTC 缓冲区 |
| `capture/device_adapters_linux.py` | D435i、DC1 和 Piper SocketCAN 适配器 |
| `inference/rtc_sink_linux.py` | 启动位姿、关节和夹爪真机控制 |
| `inference/test_model_action_linux.py` | 单次采集和推理；默认只记录完整 action，不运动 |
| `tools/check_pi05_load.py` | GPU和checkpoint加载验证 |
| `tools/hardware/check_linux_devices.py` | D435i、DC1、CAN逐项只读探针 |

特别说明：`capture/device_adapters_linux.py`是当前设备端RTC实际使用的Linux只读输入适配器，并非废弃数采文件。它最后修改于`2026-09-19 14:50:47 +08:00`，之后未再改动；当前SHA256为`d0bf574c829cdf826961e47f24921625933f76662f9353be837c13dfe19b45ca`。它已通过历史真机联调和现有离线测试，但打包前检查发现CAN启动反馈等待及夹爪输入校验仍可进一步增强；本次迁移保持该源码和导入路径不变。

模型端还依赖整个 `openpi/`，其中 H40 配置名为 `pi05_piper_finetune_h40`。迁移时不要只复制单个 OpenPI 文件。

## 2. H40固定信息

```text
config             pi05_piper_finetune_h40
checkpoint         models/pi05_piper_h40_5k/4999
action horizon     40
external action    [40, 7]
execution steps    20
flow steps         8
fps                30
task               fit the tape roll around the bottle
stats SHA256       20e6dc5df44cc0a08e5c86f048137612140bc280410aad98913234ec783e484d
```

H40配置、checkpoint、stats和客户端`--horizon 40`必须成套使用。

## 3. 每个终端的路径变量

按新机器实际位置修改第一行：

```bash
export PIPER_PROJECT_ROOT=/absolute/path/to/piper_x_real
export MODEL_PY="$HOME/.local/share/piper-x-inference/venv/bin/python"
export DEVICE_PY="$HOME/.local/share/piper-x-device/venv/bin/python"
export CHECKPOINT="$PIPER_PROJECT_ROOT/models/pi05_piper_h40_5k/4999"
export NORM_STATS="$CHECKPOINT/assets/local/piper_x_recordings_v000/norm_stats.json"
export ORBBEC_SDK_ROOT="$HOME/.local/share/piper-x-device/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37"
export PIPER_CAN_CHANNEL=can0
export PIPER_CAN_INTERFACE=socketcan
export TASK='fit the tape roll around the bottle'

cd "$PIPER_PROJECT_ROOT"
mkdir -p logs
```

检查H40 stats：

```bash
sha256sum "$NORM_STATS"
```

结果必须为第2节记录的SHA256。

## 4. 启动CAN

```bash
sudo modprobe gs_usb
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

ip -details -statistics link show can0
```

正常结果应包含：

```text
UP,LOWER_UP
can state ERROR-ACTIVE
bitrate 1000000
```

可用`candump can0`被动确认机械臂持续发送反馈，看到报文后按`Ctrl+C`退出。当前适配器不支持`restart-ms`。

## 5. 启动前只读设备检查

```bash
"$DEVICE_PY" -m tools.hardware.check_linux_devices \
  d435i --frames 30

env ORBBEC_SDK_ROOT="$ORBBEC_SDK_ROOT" \
  LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib" \
  "$DEVICE_PY" -m tools.hardware.check_linux_devices \
  dc1 --orbbec-sdk-root "$ORBBEC_SDK_ROOT" --frames 30

"$DEVICE_PY" -m tools.hardware.check_linux_devices \
  can --can-interface socketcan --can-channel can0 --frames 30
```

三项检查都只读取设备，不使能机械臂，也不发送动作。DC1只有一台时不要传`--serial`。

## 6. 终端1：启动H40模型服务

```bash
cd "$PIPER_PROJECT_ROOT"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OPENPI_DATA_HOME="$HOME/.local/share/piper-x-inference/openpi-cache" \
  "$MODEL_PY" -m inference.serve_piper_rtc \
  --config pi05_piper_finetune_h40 \
  --checkpoint "$CHECKPOINT" \
  --host 127.0.0.1 \
  --port 8000 \
  --num-steps 8 \
  --max-guidance 5
```

服务必须输出类似：

```text
RTC ready: horizon=40, steps=8, stats=20e6dc5df44c...
Serving at ws://127.0.0.1:8000
```

若显示horizon 10、CPU设备或其他stats，立即停止服务。

## 7. 终端2：单次只读action检查

```bash
cd "$PIPER_PROJECT_ROOT"

env ORBBEC_SDK_ROOT="$ORBBEC_SDK_ROOT" \
  LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib" \
  "$DEVICE_PY" -m inference.test_model_action_linux \
  --task "$TASK" \
  --host 127.0.0.1 \
  --port 8000 \
  --horizon 40 \
  --execution-steps 20 \
  --target-index 1 \
  --norm-stats "$NORM_STATS" \
  --orbbec-sdk-root "$ORBBEC_SDK_ROOT" \
  --output logs/linux_model_action_h40_once.json
```

此命令没有`--execute`，因此不会发送关节或夹爪动作。检查输出满足：

- `actions`形状为`[40,7]`；
- 数值全部有限；
- `server_metadata.horizon`为40；
- stats SHA256正确；
- 前20步的关节目标和夹爪closedness合理。

## 8. 终端2：完整RTC只读测试

```bash
env ORBBEC_SDK_ROOT="$ORBBEC_SDK_ROOT" \
  LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib" \
  "$DEVICE_PY" -m inference.run_piper_rtc_linux \
  --read-only \
  --task "$TASK" \
  --host 127.0.0.1 \
  --port 8000 \
  --horizon 40 \
  --execution-steps 20 \
  --latency-window 20 \
  --duration-s 90 \
  --response-timeout-s 180 \
  --norm-stats "$NORM_STATS" \
  --orbbec-sdk-root "$ORBBEC_SDK_ROOT" \
  --summary logs/linux_rtc_h40_s20_n8_readonly.json
```

不添加`--execute-direct`或`--execute-inverted`时使用空执行器，不改变机械臂状态。通过标准：

- 服务契约校验通过；
- 三路输入持续更新；
- 无连续超期、队列冲突或输入过期；
- 稳定推理P95小于20个控制周期，即约667 ms，并保留调度余量。

## 9. 真机RTC

只有第7、8节通过，六轴已经使能、机械臂状态正常且工作空间安全时，才使用本节。

当前执行器实际配置：

```text
启动关节位姿  [0, 0.1, -0.1, 0.6, 0, 0] rad
启动夹爪宽度  0 m（闭合）
速度            15%
固件限位容忍度  5°
target jump     已禁用
```

直接使用模型关节符号：

```bash
env ORBBEC_SDK_ROOT="$ORBBEC_SDK_ROOT" \
  LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib" \
  "$DEVICE_PY" -m inference.run_piper_rtc_linux \
  --read-only \
  --execute-direct \
  --task "$TASK" \
  --host 127.0.0.1 \
  --port 8000 \
  --horizon 40 \
  --execution-steps 20 \
  --latency-window 20 \
  --duration-s 30 \
  --response-timeout-s 180 \
  --norm-stats "$NORM_STATS" \
  --orbbec-sdk-root "$ORBBEC_SDK_ROOT" \
  --summary logs/linux_rtc_h40_s20_n8_direct.json
```

`--read-only`在这里选择真实设备输入路径；`--execute-direct`会把空执行器替换为Linux真机执行器，因此该组合会控制机械臂。

只有坐标验证已经明确要求六关节整体取反时，才把`--execute-direct`替换为`--execute-inverted`。两者不能同时使用。

真机逐条控制记录位于：

```text
直接符号  logs/linux_rtc_direct_commands.jsonl
整体取反  logs/linux_rtc_inverted_commands.jsonl
```

夹爪只进行一次换算：

```text
width_m = 0.1001 × (1 - clip(closedness, 0, 1))
```

## 10. 停止行为

- 客户端到达`--duration-s`后退出，也可按`Ctrl+C`；
- 模型服务在终端1按`Ctrl+C`停止并释放显存；
- RTC停止后不会再发送新动作；
- 当前`close()`只断开连接，不发送立即停止命令。机械臂可能继续完成最后一条已经接收的`move_j`目标，因此终端停止后仍可能运动一段时间。

## 11. 扩展到其他horizon

现有RTC调度代码已参数化，一般不需要再修改`rtc_core.py`。新horizon必须完成以下工作：

1. 在`openpi/src/openpi/training/config.py`新增唯一配置名，把模型及freeze filter中的`action_horizon`都设为`H`；
2. 使用该配置重新训练，得到与`H`匹配的完整checkpoint；不能只修改配置后加载H40权重；
3. 携带checkpoint中的`params/`、`assets/`和对应`norm_stats.json`，记录stats SHA256；
4. 模型服务改用新配置名和checkpoint路径；
5. 客户端统一改为`--horizon H --execution-steps S`，并保证`1 <= S < H`，至少为hold step保留一个位置；
6. 单次只读检查必须得到`actions.shape == [H,7]`，服务端horizon和stats哈希必须匹配；
7. 在目标GPU重新测量稳定延迟，再选择`S`和`--latency-window`。在30 Hz下，执行窗口为`S / 30`秒，应大于推理P95并留有余量；
8. 根据实测结果调整`--num-steps`。它是采样参数，与训练horizon相互独立；
9. 依次重新完成权重加载、单次只读、完整RTC只读，最后才进行真机测试。

`norm_stats`不一定因为horizon改变而改变，但部署时必须使用新训练checkpoint随附的版本，不能凭数值看似相同而混用旧文件。

## 12. 常调参数位置

优先通过启动参数修改运行参数；只有连续真机执行器的默认硬件参数需要改源码。

| 参数 | 修改位置 | 当前值或用法 |
|---|---|---|
| 模型horizon | `openpi/src/openpi/training/config.py`中H40配置的两处`action_horizon` | `40`；模型配置和freeze filter必须一致 |
| RTC horizon | 客户端命令`--horizon` | H40使用`40`，必须与服务端一致 |
| 每轮执行步数 | 客户端命令`--execution-steps` | H40使用`20`，要求`1 <= S < H` |
| 延迟统计窗口 | 客户端命令`--latency-window` | H40使用`20` |
| 推理积分步数 | 服务端命令`--num-steps` | H40使用`8` |
| guidance上限 | 服务端命令`--max-guidance` | 当前使用`5` |
| 任务提示 | 客户端命令`--task`或`TASK`变量 | 必须使用训练任务文本 |
| 运行时间 | 客户端命令`--duration-s` | `0`表示运行到`Ctrl+C` |
| 连续真机速度 | `inference/rtc_sink_linux.py`中`PiperCommandSinkLinux.__init__()`的`speed_percent` | 当前`15%`；重启客户端后生效 |
| 固件限位容忍度 | 同一构造函数的`limit_tolerance_rad` | 当前`math.radians(5.0)` |
| 启动关节位姿 | `DEFAULT_STARTUP_JOINTS_RAD` | 当前`[0, 0.1, -0.1, 0.6, 0, 0] rad` |
| 启动夹爪宽度 | `DEFAULT_STARTUP_GRIPPER_WIDTH_M` | 当前`0.0 m`，即闭合 |
| target jump | `rtc_sink_linux.py`的`send_target()`中`RTC target jump`判断 | 当前整段已注释；`max_joint_delta=0.5`不生效 |
| 关节符号 | 真机命令`--execute-direct`或`--execute-inverted` | 当前标准命令使用direct；inverted会将六关节全部取负 |
| 夹爪全开宽度 | `inference/rtc_core.py`中`RuntimeConfig.open_width_m` | 当前`0.1001 m` |
| 单次动作索引 | 单次测试命令`--target-index` | H40合法范围为`0..39` |

单次动作脚本`inference/test_model_action_linux.py`还有独立的`--speed-percent`和`--max-joint-delta`。它们只在附加`--execute`、发送一个关节目标时生效，不会改变连续RTC执行器的速度或已禁用的target jump判断。

机械臂固件上下限不是源码常量。程序启动时通过`get_joint_angle_vel_limits()`读取；源码中的`limit_tolerance_rad`只是在读取结果的上下限外增加容忍度。

## 13. Home、warmup与延迟回退

### 13.1 Home预备动作

使用`--execute-direct`或`--execute-inverted`后，真机执行器会在启动相机和推理状态机之前完成预备动作：

```text
关节目标  [0, 0.1, -0.1, 0.6, 0, 0] rad
夹爪目标  0 m（闭合）
```

六轴必须已经使能且机械臂状态正常。程序等待最大关节误差不超过`0.01 rad`、夹爪误差不超过`0.002 m`；15秒未到位则终止启动。这里的Home是项目定义的推理预备位姿，不是机械臂固件的零点回零。纯只读模式不执行此动作。

### 13.2 模型warmup与首次计划

Home完成并取得三路有效输入后，控制器依次执行：

1. `WARMUP`：发起第一次模型请求以完成JAX编译，不记录该次延迟，也不执行其action；
2. `CALIBRATION`：发起第二次请求，开始记录稳定延迟，但仍不执行action；
3. `BOOTSTRAP`：在机械臂状态保持不变时同步生成第一段动作；第一个可执行目标与当前姿态的最大差值必须不超过`0.08 rad`；
4. 提交`execution_steps`个动作后进入滚动`RTC`。

### 13.3 H40延迟回退

延迟判定使用最近`--latency-window 20`次非WARMUP推理的滚动平均值。H40当前为30 Hz、`execution_steps=20`，可用执行时间为：

```text
20 / 30 = 0.6667 s
```

当平均端到端延迟或平均模型计算时间超过约`666.7 ms`时，当前预测无法在20步执行窗口内安全衔接，控制器进入`DRAIN`并打印：

```text
回退同步启动：平均延迟不满足实时条件
```

控制器会让已经提交的动作段执行完、清空待执行计划，等待机械臂稳定后重新进入`BOOTSTRAP`，按同步方式一次生成并执行一段动作。这里的回退仍使用同一个RTC模型服务，但不再进行连续滚动衔接；它不是另一个普通推理模型。

结果超期、错过观测触发时刻、推理异常或计划无法提交也会触发相同的同步回退流程。后续滚动延迟恢复到阈值以内后，控制器会自动重新进入RTC。
