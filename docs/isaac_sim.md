# Isaac Sim 仿真移交要求

## 当前任务

第一项仿真任务是 Piper X 对一个物体执行简单抓取。目标是先形成可复现的 observation → π0.5 → action chunk → execution → next observation 闭环，再逐步加入易碎属性实验和评测。

Isaac Sim 版本、Piper X 资产和控制接口尚未在仓库中确认，因此本文定义移交包应包含的信息，不猜测版本相关 API 或数值参数。

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

大型 USD、纹理和第三方 robot 资产不直接提交；在 `assets/manifests/` 记录下载来源、版本、许可证、校验值和仓库内期望挂载位置。仓库自身编写的小型场景定义是否提交，应在确认格式和许可证后决定。

## 最小闭环接口

仿真端需要明确提供：

```text
reset(config, seed) -> initial observation
observe()           -> timestamped camera and robot observation
execute(chunk)      -> execution record
step/observe()      -> next observation and task state
finalize()          -> episode, video, events and result
```

以上是职责描述，不是已确定的 Python API。实际函数、进程边界、同步方式和 transport 在版本与运行机器确认后再实现。

## 易碎属性实验

第一版可以让场景中的目标物体具有一组可记录、可扫参的物理属性，并输出接触和任务结果。候选属性包括质量、摩擦、接触材料、刚度、阻尼或破坏相关参数，但可用项取决于锁定 Isaac Sim 版本和物理模型。

在官方能力与实验定义确认前：

- 不把任何候选参数命名为项目统一的“易碎值”；
- 不预设损伤阈值；
- 不把碰撞过滤、限速或动作裁剪描述为 Safety Layer；
- 不声称仿真损伤等价于真实物体损伤。

## 仿真验收

一次最小可复现运行应留下：配置、软件和资产版本、随机种子、任务指令、输入 observation 摘要、action chunk 执行记录、终止原因、episode、视频和评测结果。验收条件与待确认项在 [项目状态](status.md) 中维护。
