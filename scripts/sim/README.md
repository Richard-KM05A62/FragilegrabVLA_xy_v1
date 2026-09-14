# Isaac Sim 6.1.0 最小仿真入口

本目录实现当前优先的 **policy-free instrumented episode**。它加载已经准备好的 USD 场景，验证实际 Isaac Sim 版本、physics backend、stage 单位和 Piper X DOF 顺序，执行配置中的关节位置目标，并记录双 RGB、robot state、object state、contact、动作、视频和 episode manifest。它不运行 π0.5，不生成自主抓取轨迹，也不判断抓取是否成功。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run_instrumented_episode.py` | 预检配置；在 Isaac Sim 6.1.0 中运行一次脚本动作 episode。 |
| `validate_episode.py` | 不依赖 Isaac Sim，检查 episode v0.1 的引用、时间线、双相机、state/contact/action 和视频完整性。 |
| `import_piper_urdf.py` | 用 6.1.0 `URDFImporter` 将已展开的 Piper `.urdf` 转为 USD，并保存输入、输出和导入选项的校验记录。 |

实现直接使用 NVIDIA 6.1.0 公共接口：

- [`SimulationManager`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html) 单步推进物理并读取 simulation time / physics step；
- [`RenderingManager`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.core.rendering_manager/docs/index.html) 在 observation tick 单独渲染；
- [`Articulation`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/python_scripting/robots_simulation.html) 提交命名 DOF position target；
- [`JointStateSensor` / `ContactSensor`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.sensors.experimental.physics/docs/index.html) 记录带物理时间的状态和接触；
- [`CameraSensor` / `RtxCamera`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/index.html) 记录 RGB；
- [`URDFImporter`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.asset.importer.urdf/docs/index.html) 转换 Piper URDF。

## 场景必须已经包含的内容

运行器不猜测场景参数，也不在脚本里创建桌面、物体、灯光、相机或控制增益。交接的 `scene_ref` 必须已经包含：

1. 一个 USD Physics Scene；
2. Piper X articulation root，且资产中已配置适用所选 backend 的 joint drives；
3. 一个外部视角 USD Camera 和一个腕部 USD Camera；
4. 一个带 rigid body 的目标物体；
5. 位于刚体祖先下的 Isaac Contact Sensor prim；
6. 实验所需的桌面、地面、灯光、材质和初始位姿。

相机位姿、分辨率、tick rate，物体质量/摩擦/易碎候选属性，contact threshold/radius，关节 drive/stiffness/damping，physics dt、backend、seed、动作目标和成功阈值都是实验输入。它们应写入场景或本地配置，并通过场景/配置校验值追踪，不能由运行器补默认值。

v0.1 runner 要求 `max_physics_steps` 能被 observation 周期整除，并要求两路 camera tick rate 等于由 `physics_dt_s × observation_every_physics_steps` 推导出的 observation rate。这样 `.npy`、timeline 和固定帧率视频一一对应；未来确需异步相机时再按实际第二个实现扩展 sparse timeline writer。

## Piper X 资产准备

优先检查并锁定 [`agilexrobotics/piper_isaac_sim` 的 `USD/piper_x_v1.usd`](https://github.com/agilexrobotics/piper_isaac_sim/blob/master/USD/piper_x_v1.usd)、commit 和许可证。若先复用历史 `agx_datacollect_ws` 资产，可从 [`piper_x_agx_history_candidate_v0.1.yaml`](../../assets/manifests/piper_x_agx_history_candidate_v0.1.yaml) 所记 commit 取得 URDF、Xacro 和 meshes。

历史 `piper_x_description.urdf` 只有六轴机械臂；带夹爪文件是 Xacro。Isaac Sim 6.1.0 `URDFImporter` 只接收 `.urdf`，因此必须在匹配的 ROS/Xacro 环境先展开文件，并记录 Xacro/ROS 版本和展开后 SHA-256。示意命令如下，其中路径由运行者按资产挂载位置填写：

```bash
xacro <piper_x_with_gripper_description.xacro> -o <piper_x_with_gripper.urdf>
```

历史 Xacro 静态显示 `gripper_joint1/2` 是两个独立 prismatic joint；AgileX 当前公开 Xacro 还定义了 `gripper` 驱动关节和 mimic 关系。因此不得从文件名推导夹爪 DOF，也不得把两个来源的 config 混用。

随后在 Isaac Sim 安装根目录运行导入脚本。所有影响资产的选项均为必填；joint drive、target、stiffness 和 damping 只有在团队提供实际值时才传入：

```bash
./python.sh <repo>/scripts/sim/import_piper_urdf.py \
  --urdf <materialized-piper.urdf> \
  --output-dir <empty-output-directory> \
  --fix-base <true-or-false> \
  --merge-fixed-joints <true-or-false> \
  --merge-mesh <true-or-false> \
  --collision-from-visuals <true-or-false> \
  --allow-self-collision <true-or-false> \
  --run-asset-transformer <true-or-false> \
  --run-multi-physics-conversion <true-or-false> \
  --ros-package agx_arm_description=<package-directory>
```

输出目录会包含生成的 USD 和 `piper_urdf_import_manifest.json`；manifest 会列出所有生成文件的相对路径、大小和 SHA-256。将生成 USD 加入场景后，需在 6.1.0 中检查 articulation root、DOF 顺序、limits、夹爪对称运动、collision、joint drive 和静态落体/接触行为，再将生成资产的相对引用写入场景包。

## 配置和运行

复制 [`experiment_v0.1.example.yaml`](../../configs/experiment_v0.1.example.yaml) 到被忽略的 `configs/local/`，只填写场景和本次实验的真实值。instrumentation 可以保留 success/evaluation 定义为 `TBD`，此时输出的 `result.task_success` 为 `null`；运行必需字段中的 `TBD` 会被预检拒绝。

先在目标 Isaac Python 中确认 writer 依赖存在，并把输出版本保存在 smoke test 日志中：

```bash
./python.sh -c "import cv2, numpy, yaml; print(cv2.__version__, numpy.__version__, yaml.__version__)"
```

普通 Python 可以先做配置和场景路径预检，不启动 Isaac：

```bash
python3 scripts/sim/run_instrumented_episode.py \
  --config configs/local/<experiment>.yaml \
  --preflight
```

再从 Isaac Sim 6.1.0 安装根目录启动。`episode-id` 由运行者显式提供，已存在的目录不会被覆盖：

```bash
./python.sh <repo>/scripts/sim/run_instrumented_episode.py \
  --config <repo>/configs/local/<experiment>.yaml \
  --output-root <output-root> \
  --episode-id <episode-id> \
  --headless
```

完成后可在任意带 NumPy 的 Python 环境验证：

```bash
python3 scripts/sim/validate_episode.py <output-root>/<episode-id>
```

输出目录结构为：

```text
<episode-id>/
  experiment_config.yaml
  episode_manifest.json
  timeline.jsonl
  streams/base_rgb/*.npy
  streams/wrist_rgb/*.npy
  videos/base_rgb.mp4
  videos/wrist_rgb.mp4
```

失败运行保留 `episode_manifest.json`、`run_error.txt` 和 `.incomplete`；显式中断写入 `termination.status=interrupted`。完整性检查不会把两者误认为成功 episode。

## 另一台电脑最快复现

向对方交付同一个 Git commit，以及一个不含秘密的实验包：

1. 有效 experiment config、场景 USD、Piper USD、依赖资产和各自 SHA-256；
2. Piper 上游 URL/commit/license，以及 URDF import manifest；
3. Isaac Sim 6.1.0 安装来源或容器启动说明、GPU 驱动信息和实际 build 输出；
4. 场景 prim path 清单、实际 DOF 顺序、backend/device、physics dt、stage units 和视频 codec；视频 FPS 由物理步长与 observation 周期计算；
5. 一条成功的 preflight 命令、一条 episode 命令、一次通过的 validator 输出。

外部包内部使用相对 USD 引用，解压后只需修改本地 config 的包根路径。首个移交基线应固定 seed 和脚本动作，不做 domain randomization；复现通过后再把随机化参数作为显式实验输入加入。

## 后续接入 π0.5 前仍需准备

当前仿真 episode 可以作为 observation/execution/evaluation 的数据边界。接入 π0.5 仿真推理前还缺：

- 与 checkpoint 绑定的 OpenPI commit、checkpoint id、normalization stats id 和 Piper transform id；
- base/wrist 图像角色、实际 HWC shape/dtype、裁剪/缩放和缺图规则；
- Piper state 顺序以及 revolute/prismatic 单位到 OpenPI state 的精确映射；
- π0.5 action chunk 的有效维度、单位、absolute/delta 语义和反归一化；
- chunk 执行长度、控制周期、replan 条件，以及 predicted/requested/executed action 的记录规则；
- 仿真进程与推理进程的 transport、schema、超时、端口和同钟域 latency 记录；
- 一个固定 episode 的 observation transform golden sample，以及 action chunk 到命名 DOF target 的离线对照结果。

这些内容确认后，再把当前 `scripted_commands` 输入替换为 π0.5 action chunk 消费循环；本目录当前没有推理代码。
