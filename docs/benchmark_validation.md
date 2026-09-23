# FragileManip-Bench v0.1 PiPER-X 验证记录

初始验证日期：2026-09-19；LIBERO scene/camera 修正复核：2026-09-21。验证对象为 PiPER-X + 夹爪 + D435 MuJoCo 模型和当前 LIBERO-derived canonical 首任务。

## 资产与模型

- AgileX 官方 `agx_arm_sim` commit：`f8cd8b147c75d59e14f90fb0646770eefa268ed0`。
- `agx_arm_urdf` 子模块 commit：`1ee5659d02b33b9379fd647a2e5647800d67f0f4`。
- PiPER-X+D435 URDF、USD、转换脚本、相机支架与相机网格的 SHA-256 见 `assets/manifests/piper_x_official_mujoco_v0.1.yaml`。
- 编译模型：`nq=15`、`nu=7`；六轴、两指、夹爪 mimic equality 和执行器范围检查通过。
- benchmark-defined home 最大静态漂移：`1.0795e-19`；joint1 `+0.1 rad` 位置目标最终误差约 `0.001003 rad`。
- 夹爪张开位置为 `[0.05, -0.05] m`，闭合位置约为零。

## 相机

- LIBERO-derived policy 全局相机使用锁定任务的官方 `agentview`。
- 腕部相机外参由官方 `gripper_base → camera_stand → camera_link` 固定关节链加 D435 xacro 的 15 mm color offset 和 `camera_color_optical_frame` 旋转计算，不使用手填末端偏移。
- reset 帧可见夹爪两指和两指中间的目标；全局与腕部渲染均通过非恒定图像检查。
- D435 `fovy` 仍为 Candidate，等待真机内参与畸变参数绑定。

## Scripted F1 smoke

- seed：`7`；6500 个物理步；260 帧。
- `reach=true`、`pick=true`、`task_success=true`。
- 最大抬升：`0.0800508 m`。
- 峰值目标接触法向力：`0.991343 N`。
- 未检测到 slip、drop 或 unintended collision。
- 轨迹 SHA-256：`0a545bf49a272adf6c0e535bccd41d4e75ad78cd1ed137326d43ca006fd328db`。
- 运行产物仅用于本次模型核验；确认 manifest 和指标后已删除，没有作为长期 episode 归档。

## 自动检查

在 EGL 离屏渲染下，项目测试结果为 `35 passed, 2 subtests passed`。覆盖官方 commit/hash、相机关节链契约、编译模型、关节与夹爪控制、双相机渲染、PiPER-X π0.5 数据契约以及 normalization provenance。

## 当前边界

- 官方资产尚未与用户具体真机完成计量级几何、相机标定或动力学辨识。
- `piper_x_mujoco_scripted_calibration_v0.1` 可用于 zero-shot 诊断执行，但不可复用为正式微调 normalization；正式 stats 必须从实际进入 pilot 的 PiPER-X 示范重新计算。
- scripted smoke 不是 π0.5 推理或 zero-shot 抓取能力证据。
- 当前 LIBERO 黑碗是刚体 embodiment-adaptation 载体，不代表真实玻璃或陶瓷的破损阈值。
