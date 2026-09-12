# FragileGrabVLA

FragileGrabVLA 是一个面向易碎物品抓取的机器人学习研究项目，计划基于视觉语言动作模型 π0.5 和 Isaac Sim，探索仿真环境中的抓取操作与安全性问题。

## 当前阶段

项目处于早期开发阶段，近期目标是完成 π0.5 与 Isaac Sim 的集成，建立“环境观测 → 模型推理 → 动作执行 → 新观测”的最小闭环。执行链路采用 π0.5 原生的动作序列（action chunk）。

易碎物品抓取的安全性研究方案尚待确定（Open Research Question / TBD）。当前进展与研究问题见 [项目状态](docs/status.md)。

## 目录结构

项目目录规划如下，代码目录随实现逐步建立：

```text
FragileGrabVLA/
├── AGENTS.md             # 开发协作规范
├── README.md             # 项目介绍
├── configs/              # 模型、环境与任务配置
├── src/                  # 模型集成、仿真与动作执行
├── scripts/              # 运行与评估入口
├── tests/                # 自动化测试
└── docs/
    ├── architecture.md   # 系统架构
    └── status.md         # 开发进展与研究问题
```

架构与数据流见 [架构说明](docs/architecture.md)。

## 数据与运行产物

仓库维护代码、配置和项目文档。数据集、视频、模型权重、checkpoint 及 `runs/` 等运行产物保存在本地或外部存储，通过 `.gitignore` 排除；相关来源和复现说明保留在仓库中。
