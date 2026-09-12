# Architecture

## Scope

当前架构只覆盖已确认的第一阶段：在 Isaac Sim 中部署 π0.5，建立可重复运行的最小闭环操作链路。本文不定义多 VLA 框架，也不假定已有安全模块。

## Confirmed Data Flow

```text
Isaac Sim observation
        ↓
π0.5 input transform
        ↓
π0.5 inference
        ↓
action chunk
        ↓
chunk execution
        ↓
robot in Isaac Sim
        ↓
next observation
```

闭环以新的 observation 进入下一轮推理。π0.5 的原生输出是 action chunk；输入映射、动作维度、坐标系、控制频率、每次执行的 chunk 长度及重新推理条件尚未确认时，应保持显式 TBD。

## Responsibility Boundary

- 架构与设计：定义 observation、input transform、action chunk 与执行侧之间的语义和接口。
- Isaac Sim/robot 环境：提供 observation，执行动作，并返回下一时刻的环境状态。
- 双方共同确认动作空间、坐标系、单位、频率和最小闭环验收条件。

## Future Safety Extension — TBD

易碎物品安全机制目前是 **Open Research Question / TBD**，不是已确认的 Safety Layer，也不属于当前已实现架构。

若后续研究证明需要动作干预，一个候选插入位置可能位于：

```text
action chunk -> [future safety mechanism? — TBD] -> chunk execution
```

这只是扩展点标记，不代表已选择技术路线。所需信号、约束对象、干预时机、失效处理和评价指标均须经研究后再写入架构。

项目进展和未决问题见 [`status.md`](status.md)。
