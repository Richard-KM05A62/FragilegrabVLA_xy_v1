# Linux H40 RTC 推理迁移手册

本文用于把 Piper X 的 H40 pi0.5 RTC 推理系统部署到新的 Linux x86_64 机器。本文先说明环境配置；代码、权重打包以及启动和验收流程在后续章节补充。

## 1. 环境划分

系统使用两个互相隔离的 Python 3.11 环境：

| 环境 | 作用 | 是否访问硬件 |
|---|---|---|
| 模型环境 | 加载 H40 checkpoint，通过 JAX/CUDA 执行 pi0.5 推理并提供本机 WebSocket 服务 | 否 |
| 设备环境 | 读取 D435i、Orbbec DC1 和 Piper CAN，运行 RTC 调度并向机械臂发送动作 | 是 |

两个进程通过 `127.0.0.1` WebSocket 通信。分离环境可以避免 CUDA、相机 SDK 和 CAN 依赖互相影响。不要向系统 Python 安装项目依赖，也不要从原机器复制虚拟环境。

## 2. 系统要求

- Ubuntu 24.04 x86_64；
- NVIDIA GPU，建议至少 12 GB 显存；
- 可正常工作的 NVIDIA 驱动；
- Python 3.11；
- D435i、Orbbec DaBai DC1、Piper X 和 SocketCAN USB-CAN 适配器。

安装基础工具：

```bash
sudo apt update
sudo apt install -y curl git git-lfs can-utils
```

先验证 NVIDIA 驱动：

```bash
uname -m
nvidia-smi
```

`uname -m` 必须输出 `x86_64`，`nvidia-smi` 必须显示目标 GPU。若驱动失败，先修复驱动与当前 Linux 内核的匹配关系，不要继续配置模型环境，也不要用 CPU 回退代替 GPU 验收。

## 3. 统一路径

把完整项目复制到 Linux 本地目录，然后在每个新终端设置：

```bash
export PIPER_PROJECT_ROOT=/absolute/path/to/piper_x_real
export PIPER_MODEL_ENV_ROOT="$HOME/.local/share/piper-x-inference"
export PIPER_DEVICE_ENV_ROOT="$HOME/.local/share/piper-x-device"

cd "$PIPER_PROJECT_ROOT"
test -f openpi/pyproject.toml
test -f openpi/uv.lock
test -f models/pi05_piper_h40_5k/4999/params/manifest.ocdbt
```

`PIPER_PROJECT_ROOT` 必须替换成新机器上的实际绝对路径。虚拟环境和缓存放在 Linux 家目录，避免在 NTFS、移动硬盘或共享目录中创建环境。

## 4. 安装 uv

uv 只负责创建和同步隔离环境，不替代 NVIDIA 驱动：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version
```

后续所有命令使用同一个 `uv`。不要同时对同一虚拟环境运行多个同步进程。

## 5. 模型环境

### 5.1 作用

模型环境包含 OpenPI、JAX CUDA、Flax、Orbax、PyTorch 和 tokenizer 依赖，只负责加载权重和生成动作，不直接导入相机或 CAN SDK。

### 5.2 创建

```bash
cd "$PIPER_PROJECT_ROOT"
unset PYTHONPATH LD_LIBRARY_PATH

export UV_PROJECT_ENVIRONMENT="$PIPER_MODEL_ENV_ROOT/venv"
export UV_CACHE_DIR="$PIPER_MODEL_ENV_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PIPER_MODEL_ENV_ROOT/python"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export GIT_LFS_SKIP_SMUDGE=1

mkdir -p "$PIPER_MODEL_ENV_ROOT" logs

uv sync \
  --project openpi \
  --frozen \
  --python 3.11 \
  --no-dev
```

`--frozen` 强制使用随项目交付的 `openpi/uv.lock`，不能在迁移时执行依赖升级。`UV_LINK_MODE=copy` 避免项目目录与 Linux 家目录位于不同文件系统时链接失败。

当前已验证的核心版本为：Python 3.11.16、JAX/JAXLIB 0.5.3、Flax 0.10.2、Orbax 0.11.13、NumPy 1.26.4、Torch 2.7.1 和 Transformers 4.53.2。实际安装版本应由锁文件决定。

### 5.3 GPU 验收

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME="$PIPER_MODEL_ENV_ROOT/openpi-cache"

"$PIPER_MODEL_ENV_ROOT/venv/bin/python" - <<'PY'
import jax
import jax.numpy as jnp

print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu"
x = jnp.ones((256, 256), dtype=jnp.bfloat16)
y = jax.jit(lambda a: a @ a)(x)
y.block_until_ready()
print("GPU_SMOKE_OK", y.shape, y.dtype)
PY
```

必须看到 `backend: gpu`、NVIDIA `CudaDevice` 和 `GPU_SMOKE_OK`。设置 `JAX_PLATFORMS=cuda` 后，CUDA 不可用会直接报错，避免静默使用 CPU。

首次加载模型可能需要 `big_vision/paligemma_tokenizer.model`。离线迁移时应把原机器的 `openpi-cache/big_vision/paligemma_tokenizer.model` 一并复制到新机器相同缓存结构。

## 6. 设备环境

### 6.1 作用

设备环境负责两台相机、Piper CAN、RTC 调度和 WebSocket 客户端。它不加载模型权重，因此不需要安装 JAX CUDA。

### 6.2 创建

```bash
cd "$PIPER_PROJECT_ROOT"
mkdir -p "$PIPER_DEVICE_ENV_ROOT"

uv venv --python 3.11 "$PIPER_DEVICE_ENV_ROOT/venv"

uv pip install --python "$PIPER_DEVICE_ENV_ROOT/venv/bin/python" \
  "numpy==1.26.4" \
  "python-can==4.6.1" \
  "pyrealsense2==2.58.4.10922" \
  "websockets==17.1" \
  "git+https://github.com/agilexrobotics/pyAgxArm.git@cc498c00af0bcb9e297943e94f4792c0e3ee5b2c"

uv pip install --python "$PIPER_DEVICE_ENV_ROOT/venv/bin/python" \
  -e "$PIPER_PROJECT_ROOT/openpi/packages/openpi-client"
```

导入验收：

```bash
"$PIPER_DEVICE_ENV_ROOT/venv/bin/python" - <<'PY'
import numpy
import can
import pyrealsense2
import websockets
import pyAgxArm
import openpi_client
print("DEVICE_IMPORTS_OK")
PY
```

## 7. Orbbec SDK 与相机权限

### 7.1 作用

D435i 使用 `pyrealsense2`；DC1 使用 Orbbec Linux C SDK。Windows 的 DLL 和 Windows SDK 不能用于 Linux。

将 **Linux x64 Orbbec SDK v1.10.37** 解压到：

```text
$PIPER_DEVICE_ENV_ROOT/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37
```

配置并检查动态库：

```bash
export ORBBEC_SDK_ROOT="$PIPER_DEVICE_ENV_ROOT/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37"
export LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$ORBBEC_SDK_ROOT/SDK/lib/libOrbbecSDK.so"
ldd "$ORBBEC_SDK_ROOT/SDK/lib/libOrbbecSDK.so"
```

安装项目提供的相机 udev 规则，使普通用户可以访问 USB 相机：

```bash
cd "$PIPER_PROJECT_ROOT"
sudo install -m 0644 tools/hardware/rules_linux/99-obsensor-libusb.rules /etc/udev/rules.d/
sudo install -m 0644 tools/hardware/rules_linux/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

随后重新插拔两台相机。当前 DC1 的 SDK 序列号为空，只连接一台 Orbbec 相机时不要设置 DC1 序列号；USB 描述符中的序列号不能代替 SDK 序列号。

## 8. CAN 环境

### 8.1 作用

`gs_usb` 把 USB-CAN 适配器注册为 Linux `can0`；SocketCAN 以 1 Mbps 接收机械臂状态并发送控制帧。

每次开机或重新连接适配器后执行：

```bash
sudo modprobe gs_usb
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

export PIPER_CAN_CHANNEL=can0
export PIPER_CAN_INTERFACE=socketcan
```

检查：

```bash
ip -details -statistics link show can0
candump can0
```

正常状态应包含 `UP,LOWER_UP`、`ERROR-ACTIVE` 和 `bitrate 1000000`，`candump` 应持续收到反馈帧。看到报文后按 `Ctrl+C` 退出。当前适配器不支持 `restart-ms`，不要配置 Bus-Off 自动重启参数。

## 9. 环境完成标准

进入模型和硬件测试前，必须同时满足：

1. `nvidia-smi` 正常；
2. 模型环境输出 `GPU_SMOKE_OK`，JAX 后端为 GPU；
3. 设备环境输出 `DEVICE_IMPORTS_OK`；
4. `libOrbbecSDK.so` 的 `ldd` 结果没有 `not found`；
5. 两份相机 udev 规则已安装并重新插拔设备；
6. `can0` 为 1 Mbps、`ERROR-ACTIVE`，能够收到 Piper CAN 帧；
7. H40 checkpoint 和 stats 文件存在且未被修改。

这些检查只证明运行环境完整。完成相机取帧、模型权重加载和只读 RTC 验收之前，不运行真机控制模式。
