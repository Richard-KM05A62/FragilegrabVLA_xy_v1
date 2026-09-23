# FragileGrabVLA

FragileGrabVLA 是一个围绕 Piper X 与视觉语言动作模型 π0.5 的易碎物体抓取研究项目。仓库用于维护真实机器人数据采集、数据处理、MuJoCo benchmark 与保留的 Isaac Sim 仿真验证、策略执行、实验记录和统一评测。

## 当前研究范围

- Policy：已跑通 `pi05_base` 双 RGB、七维 PiPER-X 接口和 MuJoCo 闭环；当前 scripted calibration stats 仅用于 zero-shot 诊断，正式微调 normalization 待从训练示范重算
- Robot：真机 Piper X；canonical MuJoCo 模型为 AgileX 官方 `agx_arm_sim` 的 PiPER-X + 夹爪 + D435 锁定资产
- Simulation：MuJoCo 主线；canonical benchmark 保留 LIBERO 的 arena / furniture / camera / placement logic，替换为 PiPER-X 和 fragile / fragile-like target，并重新定义 task predicate；safety stress-test 只在该 canonical scene 上生成可追溯变体；Isaac Sim 保留支线
- Sensors：真机计划外部 RGB-D 与腕部相机；当前 MuJoCo 同时输入并记录全局 RGB 与腕部 RGB
- Task：从简单抓取开始，逐步研究易碎属性、损伤判定和安全性问题

当前不引入 OpenVLA、其他 VLA、多策略注册表或多机器人抽象。π0.5 保留原生 action chunk。易碎物体安全机制仍是开放研究问题，项目尚无已确认的 Safety Layer。

## 当前状态

已加入 FragileManip-Bench v0.1 的 Qualification（reach/pick/transport/place）与 F1 Gentle Lift；F2–F5 仅为规格。PiPER-X 低层控制、双相机、OpenPI 接口和 LIBERO-Spatial 黑碗到盘子场景已跑通。当前 zero-shot 只显示弱接近趋势且过早闭合；未进入 pre-grasp、未接触、未抬升，不得表述为抓取成功。

仓库已有一套 Piper X 双相机只读数据采集代码，包含主机时钟近邻对齐和 LeRobot 写盘逻辑。代码存在不等于完整链路已经验证；当前验证状态和风险见 [项目状态](docs/status.md)。

Isaac Sim 6.1.0 的 policy-free 入口仍保留为独立支线，但不是当前任务前置。近期主线是：完成 zero-shot 诊断对照，在锁定 LIBERO-derived 场景采集 PiPER-X 完整抓放示范，从实际训练集重算 normalization，然后做 10k–20k step embodiment-adaptation pilot。

## 研究链路

```text
Real Robot Data Collection / MuJoCo Benchmark / Isaac Sim Recording
                      ↓
             Versioned Episode Data
                      ↓
          Validation and Data Processing
                      ↓
              π0.5 Training/Inference
                      ↓
            MuJoCo / Isaac Sim / Piper X Execution
                      ↓
       Experiment Recording and Evaluation
```

## 仓库入口

- [FragileManip-Bench v0.1 运行手册](docs/benchmark.md)
- [π0.5–PiPER-X 仿真诊断](docs/simulation_diagnostics.md)
- [系统架构](docs/architecture.md)
- [项目状态](docs/status.md)
- [真实数据采集](docs/data_collection.md)
- [真实 PiPER-X → 本地 MuJoCo 遥操作采集](docs/piper_mujoco_teleoperation.md)
- [数据格式](docs/data_format.md)
- [π0.5 接入约束](docs/pi05.md)
- [PiPER π0.5 运行手册](scripts/policy/README.md)
- [Isaac Sim 复现要求](docs/isaac_sim.md)
- [Isaac Sim 6.1.0 运行入口](scripts/sim/README.md)
- [目标机 Codex 执行任务](docs/handoffs/target_machine_codex_isaac_pi05_v0.1.md)
- [评测设计](docs/evaluation.md)
- [研究问题](docs/research_questions.md)

实际运行前，应先确认对应设备、SDK、OpenPI 和 Isaac Sim 版本。具体命令、机器路径和实验参数不写在本文件中。

## 开始工作

1. 先阅读 [项目状态](docs/status.md)，确认所需能力属于 Verified、In Progress 还是 TBD。
2. 当前 MuJoCo 主线从 [benchmark 运行手册](docs/benchmark.md) 和 [π0.5–PiPER-X 诊断](docs/simulation_diagnostics.md) 开始；真机数采见 [真实数据采集](docs/data_collection.md)，Isaac 支线另见 [Isaac Sim 复现要求](docs/isaac_sim.md)。
3. 在运行机器上锁定依赖和外部资产版本，把个人设备与路径写入不提交的本地配置。
4. 将运行使用的配置、代码版本、数据或场景版本和产物引用写入 episode manifest。

## 数据与产物

代码、配置模板、数据规范、外部资源 manifest 和可复现实验说明进入 Git。数据集、视频、模型权重、checkpoint、运行日志、生成资产和本机配置保存在本地或外部存储，不提交到仓库。
