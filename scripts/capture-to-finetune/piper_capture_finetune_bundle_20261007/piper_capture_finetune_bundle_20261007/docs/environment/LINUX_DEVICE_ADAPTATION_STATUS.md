# Linux 设备适配实现状态

更新：2026-09-19。本文件记录旁路实现；原 Windows 文件保持不变。本阶段仅实现只读输入，不包含机械臂使能或运动控制。

## 新增文件

| 文件 | 用途 |
|---|---|
| `capture/device_adapters_linux.py` | Linux D435i、DaBai DC1 和 Piper CAN 只读适配器 |
| `inference/rtc_sources_linux.py` | RTC harness 的 Linux 三路输入源 |
| `inference/run_piper_rtc_linux.py` | Linux 专用 RTC 入口，复用原调度并替换输入源 |
| `tools/hardware/check_linux_devices.py` | 一次只测试一个设备的只读探针 |
| `tests/test_device_adapters_linux.py` | 无硬件路径/config 测试 |

未修改：`capture/aligned_capture.py`、`inference/rtc_sources.py`、`inference/run_piper_rtc.py`。

## 已实现

- DC1 从 `ORBBEC_SDK_ROOT` 或参数查找 `libOrbbecSDK.so`，通过 `ctypes.CDLL(..., RTLD_GLOBAL)` 加载。
- 保留原 C API 签名、RGB buffer 独立复制以及 color frame/frameset 双重释放。
- D435i 使用 Linux `pyrealsense2`，仍优先 RGB8，必要时从 BGR8 转成独立 RGB 数组。
- Piper CAN 的 channel/interface 可由构造参数或 `PIPER_CAN_CHANNEL`、`PIPER_CAN_INTERFACE` 指定。
- Piper 类只调用 connect、状态 getter 和 disconnect；不调用 enable 或动作接口。
- Linux RTC 入口继续使用原来的 `NullCommandSink`。

## 环境和资源

已建立独立设备环境：

```text
/home/bac-alive/.local/share/piper-x-device/venv
```

已安装并通过导入：pyAgxArm 1.0.0（官方 commit `cc498c00`）、python-can 4.6.1、pyrealsense2 2.58.4.10922、openpi-client 0.1.0、websockets 17.1 和 NumPy 1.26.4。GPU 模型环境没有被修改。

与原 Windows SDK 对应的 Orbbec v1.10.37 Linux x64 SDK 已放在：

```text
/home/bac-alive/.local/share/piper-x-device/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37
```

ZIP SHA256 为 `0a8921e659ef172a535ec1631379d29c2542ad3afabe2b5592d20be1db0158c8`。`libOrbbecSDK.so` 及依赖已通过动态加载和 `ldd` 检查。

两份相机 udev 规则已经安装到 `/etc/udev/rules.d/`，其 SHA256 与项目中的规则一致：

- Orbbec：`04bd6ef1e10cb7cf5ab15d78d6ef00987ca5412ee99a5f1a6b1a5bba760fb473`
- RealSense：`34c287000862710b60732aea942bda6f51faf7436b5b53f66fbfd021bc8c22dd`

## 真机验证结果

2026-09-19 已识别并验证两台相机：

| 设备 | USB/序列号 | 单机连续取帧 | 关闭后重连 | 双机并发 |
|---|---|---:|---:|---:|
| Intel RealSense D435i | `8086:0b3a` / `261822072150`，USB 3.2 | 300 帧，29.982 FPS，最大帧间隔 34.609 ms | 30 帧，30.033 FPS | 300 帧，29.981 FPS |
| Orbbec DaBai DC1 | `2bc5:0557`、`2bc5:0657` / USB 描述符 `CC1N16201XW`，当前 USB 2.0 | 300 帧，29.910 FPS，最大帧间隔 37.372 ms | 30 帧，29.941 FPS | 300 帧，29.911 FPS |

两台相机均返回 `(480, 640, 3)`、`uint8`、独立持有内存的 RGB 数组。并发运行期间未见掉流或 SDK 冲突。测试日志：

- `logs/linux_d435i_probe.log`
- `logs/linux_dc1_probe.log`
- `logs/linux_dual_camera_probe.log`

Orbbec SDK v1.10.37 对这台 DC1 返回空 SDK 序列号。USB 描述符中的 `CC1N16201XW` 不能用于 SDK 的按序列号选择；仅连接一台 Orbbec 相机时应省略 `--serial`。Linux 适配器已增加预检查，无效序列号会安全报错，不会再进入原生 SDK 崩溃路径。

Piper CAN 已完成真机验证：

- USB-CAN 为 `1d50:606f` Geschwister Schneider CAN adapter，内核驱动 `gs_usb`，接口 `can0`。
- `can0` 运行在 1,000,000 bit/s，状态 `ERROR-ACTIVE`；Bus-Off、总线错误、仲裁丢失均为 0。
- 5 秒被动监听收到 12,106 帧，包含 `0x1C0`、`0x251`—`0x266`、`0x2A1`—`0x2A8` 等反馈 ID。
- pyAgxArm 读取 300 次有效的 7 维状态；断开重连后，100 次关节及夹爪 SDK 时间戳均持续更新。
- 机械臂开始测试前六个关节已经处于使能状态；本实现未调用使能或运动接口。
- SDK 报告主状态 `NORMAL`，关节限位和通信错误标志均为 false；`get_firmware()` 在当前设备上返回 `None`。

三路联合采集也已通过：D435i、DC1 和 Piper CAN 同时进入 RTC `DataBuffer`，成功产生完整 observation，输入年龄分别约为 31.5 ms、28.1 ms 和 2.4 ms，停止后没有残留采集线程。

CAN 与联合测试日志：

- `logs/linux_can_passive.log`
- `logs/linux_piper_can_probe.log`
- `logs/linux_piper_can_reconnect.log`
- `logs/linux_live_sources_probe.log`

相机和 CAN 输入适配均已通过真机验证。完整的“采集—pi0.5 RTC 推理—空命令接收器”链路也已通过；实测数据与当前性能限制见 `LINUX_INFERENCE_ENVIRONMENT.md` 的“实际只读推理验证”。

设备依赖应安装到实际运行输入 harness 的 Python 环境。GPU 模型服务仍可使用独立模型环境，通过 `127.0.0.1` WebSocket 通信。

相机 udev 规则的已执行安装命令如下，留作重装记录：

```bash
sudo install -m 0644 tools/hardware/rules_linux/99-obsensor-libusb.rules /etc/udev/rules.d/
sudo install -m 0644 tools/hardware/rules_linux/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

CAN 接入后先检查 `ip -details link show`，再按 pyAgxArm 官方脚本或下面等价命令激活：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

## 只读验证命令

相机探针已执行通过，以下命令保留用于复测：

```bash
source /home/bac-alive/.local/share/piper-x-device/venv/bin/activate

python -m tools.hardware.check_linux_devices d435i --serial 261822072150 --frames 300

ORBBEC_SDK_ROOT=/home/bac-alive/.local/share/piper-x-device/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37 \
LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python -m tools.hardware.check_linux_devices dc1 --frames 300

PIPER_CAN_INTERFACE=socketcan \
PIPER_CAN_CHANNEL=can0 \
python -m tools.hardware.check_linux_devices can --frames 300
```

三项分别通过后，才运行 Linux RTC 只读入口：

```bash
ORBBEC_SDK_ROOT=/home/bac-alive/.local/share/piper-x-device/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37 \
LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PIPER_CAN_INTERFACE=socketcan \
PIPER_CAN_CHANNEL=can0 \
python -m inference.run_piper_rtc_linux --read-only \
  --task '<training task prompt>' --host 127.0.0.1 --port 8000
```

该入口会连接设备并读取状态，但仍不会使能或控制机械臂。
