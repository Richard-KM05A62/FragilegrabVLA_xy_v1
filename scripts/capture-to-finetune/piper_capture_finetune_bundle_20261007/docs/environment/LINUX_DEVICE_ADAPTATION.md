# Linux 真机推理：Windows 依赖适配清单

更新：2026-09-19。本文件只概括迁移范围，不表示这些设备已在 Linux 验证，也不涉及机械臂运动控制。

## 必须适配

1. **DaBai DC1 / Orbbec**
   - 当前 `capture/aligned_capture.py` 写死 Windows SDK 目录，加载 `OrbbecSDK.dll`，并调用 `os.add_dll_directory()`、`ctypes.WinDLL()`。
   - Linux 需要同版本、同架构的 Orbbec Linux SDK，加载 `libOrbbecSDK.so`，改用 `ctypes.CDLL()`；通过 `--orbbec-sdk-root` 或 `ORBBEC_SDK_ROOT` 指定位置。
   - 安装 SDK 自带的 USB udev 规则，并保证普通用户可访问相机。
   - 现有 C API、RGB buffer 复制及 `color + frameset` 双重释放逻辑应保留，避免帧缓冲泄漏。

2. **Piper CAN / pyAgxArm**
   - 当前配置固定为 `comm="can"`、`channel="0"`、`interface="agx_cando"`，此前只在 Windows 环境验证。
   - Linux 需要安装官方 SDK 的 Linux 依赖及 CAN 驱动，确认该 interface 名称是否仍适用，并配置设备权限。
   - 先只验证 connect、状态读取和 disconnect；不得调用 enable 或运动接口。真正控制的 `CommandSink` 尚未实现。

3. **模块解耦**
   - `inference/rtc_sources.py` 当前从 `capture/aligned_capture.py` 导入相机和 CAN 类；后者在模块顶层还导入 LeRobot、Datasets、PyAV 和 pyAgxArm。
   - 建议把两台相机适配器拆到独立的 `devices/cameras.py`，让数采和推理共同使用。这样能单独测试相机，也避免纯推理输入层被无关的数据集依赖阻塞。

## 主要是环境配置

- **D435i**：代码使用跨平台 `pyrealsense2`，通常无需重写；Linux 需要 librealsense、Python binding、RealSense udev 规则和 USB 权限。
- **路径与虚拟环境**：删除 Windows 盘符、Windows venv 和 Downloads SDK 默认路径；Linux 模型端统一使用 `openpi/.venv/bin/python`，设备依赖也必须装入实际运行 harness 的解释器。
- **动态库查找**：必要时配置 `LD_LIBRARY_PATH`，不能复制或使用 Windows DLL。
- **时间戳**：现有 `time.monotonic_ns()` 在 Linux 可继续使用，不需要修改对齐算法。

## 推荐验证顺序

1. D435i 单独连续读取、退出和重连。
2. DC1 单独连续读取，检查 RGB、帧率、内存和资源释放。
3. 双相机同时运行。
4. Piper CAN 只读状态。
5. 三路输入进入 RTC buffer，但使用 `NullCommandSink`。
6. 最后才设计、审查和测试真实运动接口；任何运动测试前先通知用户。

每台相机都应验证：序列号选择、RGB `uint8[H,W,3]`、30 Hz、单调电脑时间戳、Ctrl+C 释放、长时间运行内存稳定。

## 本地 WebSocket 推理结构

当前结构是同一台 Linux 机器上的两个进程：GPU 模型服务负责 π0.5/RTC，实时 harness 负责相机、CAN、缓存和调度，二者通过 `127.0.0.1` WebSocket 通信。这是 OpenPI 风格的正常部署方式，便于隔离 GPU 环境与设备生命周期；回环通信延迟也会被当前端到端延迟估计包含。

它不是唯一方案。如果后续确认 WebSocket 序列化、进程切换或故障恢复成为实时瓶颈，可以改成同进程直接调用 policy；但第一版优先保留当前结构，更容易独立测试和定位问题。服务只监听回环地址，不应无认证暴露公网。
