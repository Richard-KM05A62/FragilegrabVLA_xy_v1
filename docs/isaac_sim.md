# Isaac Sim 仿真移交要求

## 版本基线

项目成员已选择 **NVIDIA Isaac Sim 6.1.0** 作为当前仿真目标版本。[NVIDIA 6.1.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html) 和[官方下载页](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/download.html)是版本依据；本仓库尚无目标机器安装、启动日志或 smoke test，因此该选择是 Confirmed，不是运行 Verified。

6.1.0 中旧的 `isaacsim.sensors.physics` 已弃用，新的仿真 instrumentation 应依据官方 [`isaacsim.sensors.experimental.physics`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.sensors.experimental.physics/docs/index.html) 公共接口。该扩展可提供带 simulation time、physics step 和 validity 的 contact、joint state 与 effort 读数。physics backend 尚未确定；尤其 effort 能力可能随 backend 不同，因此必须作为配置与运行证据记录。

## 当前任务

第一项仿真任务是 Piper X 对一个物体执行简单抓取。目标是先形成可复现的 observation → π0.5 → action chunk → execution → next observation 闭环，再逐步加入易碎属性实验和评测。

当前优先顺序是先在 6.1.0 中形成不依赖 π0.5 的 instrumented episode，验证 robot/object state、双相机角色、动作记录、接触事件、终止和写盘语义；随后再接入 π0.5 action chunk。仓库已实现 [`scripts/sim/run_instrumented_episode.py`](../scripts/sim/run_instrumented_episode.py)、[`import_piper_urdf.py`](../scripts/sim/import_piper_urdf.py) 和 [`validate_episode.py`](../scripts/sim/validate_episode.py)，但目标机器尚未提供运行证据。Piper X 最终资产 commit、控制增益和实验参数继续保持 TBD。

AgileX 官方公开仓库 [`agilexrobotics/piper_isaac_sim`](https://github.com/agilexrobotics/piper_isaac_sim) 提供 [`USD/piper_x_v1.usd`](https://github.com/agilexrobotics/piper_isaac_sim/blob/master/USD/piper_x_v1.usd) 与 Piper X description，[`agilexrobotics/agx_arm_urdf`](https://github.com/agilexrobotics/agx_arm_urdf) 提供 Piper X URDF/Xacro 与 mesh。两者可作为 Candidate 上游；采用前必须锁定具体 commit，记录许可证、校验值、joint/mesh 引用和 Isaac Sim 6.1.0 导入结果。仓库当前不复制这些大型第三方资产。

场景基线不从空白 USD 自行搭建。优先复用并迁移已有 Piper 仿真任务：

1. AgileX College 的 [`IsaacLab_Data_Collection`](https://github.com/agilexrobotics/Agilex-College/tree/master/isaac_sim/agx_arm_IsaacLab/IsaacLab_Data_Collection) 已提供 `Isaac-Stack-Cube-Piper-IK-Rel-v0` 方块堆叠环境、遥操作、自动采集和回放入口，可作为简单抓取/堆叠场景与 IK 控制基线；其声明环境是 Isaac Sim 5.1.0.0、IsaacLab 0.54.3，必须先做 6.1.0 迁移验证。
2. DynamicVLA 的[官方仓库](https://github.com/hzxie/DynamicVLA)提供可下载的 DOM USD scenes/objects、Piper `pick`/`place`/`long-horizon` 仿真入口和分离的 evaluation/inference 进程，可作为 VLA 场景、任务终止、视频评测和本机进程通信参考；其公开基线是 Isaac Sim 4.5.0、Isaac Lab 2.2.1，且资产和代码许可证需分别核对，不能直接声称兼容 6.1.0。

以上游任务包为起点，把其中 Piper 机器人资产替换或核对为 AgileX 官方 Piper X USD；只添加本项目必需且上游缺少的第二相机、contact/effort 记录和 episode instrumentation。任何 drive、物体属性、相机位姿或控制参数都优先继承所选上游版本并记录来源；版本迁移中必须修改的值通过差异清单和 6.1.0 运行证据确认，不凭视觉效果手工调参。

HybridVLA 和 DexVLA 可用于核对 VLA 的多相机排列、robot state 归一化、action chunk/队列和定期重推理方式，但当前公开实现不提供 Piper X + Isaac Sim 场景：HybridVLA 的公开仿真评测基于 RLBench/CoppeliaSim；DexVLA 的 `smart_eval_agilex.py` 提供三相机、状态统计和 action queue 示例，但仓库内默认 AgileX environment 是待替换的 fake environment。因此它们不是本项目的 USD 来源，也不作为 π0.5 模型接口依赖。

历史 AGX workspace 的夹爪 Xacro 与 AgileX 当前公开 Xacro 不是同一个 joint contract：前者静态检查得到两个独立的 `gripper_joint1/2`，当前公开文件还包含一个 `gripper` 驱动关节和 mimic 关系。两者不能共用 DOF 列表或控制 target。有效 config 必须填写最终 USD 的实际 DOF 顺序，运行器会逐项比对。

## 可移交场景包

队友收到仓库内容和外部资产后，应能按说明在锁定版本的 Isaac Sim 中复现。场景包至少包括：

- Isaac Sim、Isaac Lab 或扩展的精确版本和安装来源；
- Piper X robot 资产来源、commit 或版本、许可证与校验值；
- 场景文件以及 robot、桌面、物体、灯光和相机的相对资产引用；
- 外部相机和腕部相机的角色、安装关系、分辨率与输出字段；
- articulation、关节顺序、夹爪、驱动和控制接口说明；
- task 初始状态、reset、随机种子、成功与终止条件；
- 运行配置、入口、日志、视频和 episode 输出位置；
- 一次已通过的 smoke test 记录。

大型 USD、纹理和第三方 robot 资产不直接提交；在 `assets/manifests/` 记录下载来源、版本、许可证、校验值和仓库内期望挂载位置。历史 AGX workspace 中的 Piper X 文件已由 [`piper_x_agx_history_candidate_v0.1.yaml`](../assets/manifests/piper_x_agx_history_candidate_v0.1.yaml) 记录为 Candidate，其带夹爪输入仍需从 Xacro 展开、经 6.1.0 导入并现场核对。

## 已实现的最小闭环接口

仿真端需要明确提供：

```text
load scene/config   -> validate version, backend, units, prims and DOFs
warmup              -> valid camera, robot and contact readings
execute(target)     -> requested/executed named-DOF position target record
physics/render      -> timestamped camera, robot, object and contact observation
finalize            -> episode manifest, timeline, RGB arrays and two videos
```

当前动作来源固定为 config 中的 `scripted_joint_position_targets`。运行器在每个指定的 episode physics step 之前提交 target，之后通过 `SimulationManager.step()` 推进一个物理步，仅在 observation tick 调用 `RenderingManager.render()`。π0.5 transport、action chunk 消费和在线 task termination 尚未实现。

运行器只加载从上述上游任务包选择、锁定并完成 6.1.0 迁移验证的场景，不在核心逻辑中生成桌面、物体、相机、contact threshold 或 drive gain。这些值来自上游场景或有来源记录的 instrumentation 差异，并进入有效场景和 `configs/local/` 配置。完整入口、输出布局和跨电脑复现步骤见 [`scripts/sim/README.md`](../scripts/sim/README.md)。

## 6.1.0 instrumentation 顺序

在接入 π0.5 前，最小仿真适配按以下证据顺序推进：

1. 在目标机器补齐 config，运行 preflight；
2. 记录 Isaac Sim 6.1.0 build、启动方式和实际 physics backend；
3. 锁定 Piper X asset commit/import manifest，并核对 articulation、DOF 顺序、limits、单位、夹爪、drive 和许可证；
4. 验证固定 seed/scene 后 robot/object 初始状态可重复读取；
5. 验证外部与腕部两个相机的实际 shape、dtype、时间和视频；
6. 验证 requested/executed action、robot state、object state、有效 contact 和 joint effort；
7. 输出并通过 `fragilegrab.episode/0.1` 完整性检查，再进入 π0.5 transform 与 inference 适配。

contact sensor 的 threshold、采样周期和检测半径均为实验参数，当前不得写默认值。第一轮可以验证 raw contact record 的可用性，但不得把任意接触直接命名为 damage 或 safety violation。

## 易碎属性实验

第一版可以让场景中的目标物体具有一组可记录、可扫参的物理属性，并输出接触和任务结果。候选属性包括质量、摩擦、接触材料、刚度、阻尼或破坏相关参数，但可用项取决于锁定 Isaac Sim 版本和物理模型。

在官方能力与实验定义确认前：

- 不把任何候选参数命名为项目统一的“易碎值”；
- 不预设损伤阈值；
- 不把碰撞过滤、限速或动作裁剪描述为 Safety Layer；
- 不声称仿真损伤等价于真实物体损伤。

## 仿真验收

一次最小可复现运行应留下：配置、软件和资产版本、随机种子、任务指令、输入 observation、脚本 target 执行记录、终止原因、episode 和双视角视频。由于当前不运行 π0.5 且成功定义仍为 TBD，`result.task_success` 和 safety evaluation 保留 `null`。验收条件与待确认项在 [项目状态](status.md) 中维护。
