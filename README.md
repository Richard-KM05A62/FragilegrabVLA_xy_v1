# FragileGrabVLA

FragileGrabVLA 是一个围绕 Piper X 与视觉语言动作模型 π0.5 的易碎物体抓取研究项目。仓库用于维护真实机器人数据采集、数据处理、Isaac Sim 仿真验证、策略执行、实验记录和统一评测。

## 当前研究范围

- Policy：π0.5
- Robot：Piper X
- Simulation：Isaac Sim
- Sensors：外部 RGB-D 相机与腕部相机
- Task：从简单抓取开始，逐步研究易碎属性、损伤判定和安全性问题

当前不引入 OpenVLA、其他 VLA、多策略注册表或多机器人抽象。π0.5 保留原生 action chunk。易碎物体安全机制仍是开放研究问题，项目尚无已确认的 Safety Layer。

## 当前状态

仓库已有一套 Piper X 双相机只读数据采集代码，包含主机时钟近邻对齐和 LeRobot 写盘逻辑。代码存在不等于完整链路已经验证；当前验证状态和风险见 [项目状态](docs/status.md)。

Isaac Sim 场景、Piper X 仿真执行、π0.5 Piper 数据适配和 evaluation 仍处于设计或待实现阶段。

## 研究链路

```text
Real Robot Data Collection / Isaac Sim Recording
                      ↓
             Versioned Episode Data
                      ↓
          Validation and Data Processing
                      ↓
              π0.5 Training/Inference
                      ↓
            Isaac Sim / Piper X Execution
                      ↓
       Experiment Recording and Evaluation
```

## 仓库入口

- [系统架构](docs/architecture.md)
- [项目状态](docs/status.md)
- [真实数据采集](docs/data_collection.md)
- [数据格式](docs/data_format.md)
- [π0.5 接入约束](docs/pi05.md)
- [Isaac Sim 复现要求](docs/isaac_sim.md)
- [评测设计](docs/evaluation.md)
- [研究问题](docs/research_questions.md)

实际运行前，应先确认对应设备、SDK、OpenPI 和 Isaac Sim 版本。具体命令、机器路径和实验参数不写在本文件中。

## 开始工作

1. 先阅读 [项目状态](docs/status.md)，确认所需能力属于 Verified、In Progress 还是 TBD。
2. 真机数采从 [真实数据采集](docs/data_collection.md) 开始；仿真移交从 [Isaac Sim 复现要求](docs/isaac_sim.md) 开始。
3. 在运行机器上锁定依赖和外部资产版本，把个人设备与路径写入不提交的本地配置。
4. 将运行使用的配置、代码版本、数据或场景版本和产物引用写入 episode manifest。

## 数据与产物

代码、配置模板、数据规范、外部资源 manifest 和可复现实验说明进入 Git。数据集、视频、模型权重、checkpoint、运行日志、生成资产和本机配置保存在本地或外部存储，不提交到仓库。
