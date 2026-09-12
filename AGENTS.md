# FragileGrabVLA 开发规则

## 开始改动前

开始任何代码、配置或文档改动前，必须先阅读：

- `README.md`
- `docs/architecture.md`
- `docs/status.md`

以 GitHub 仓库中的代码和文档作为项目的长期事实来源。聊天记录可以辅助讨论，但未写入仓库的内容不应被视为已确认决策。

## 当前范围

- 课题是易碎物品抓取。
- 当前仅计划使用 π0.5，近期目标是将其部署到 Isaac Sim 并打通最小闭环。
- 保留 π0.5 原生 action chunk 范式，不得未经确认改造成单步动作或通用模型输出。
- 不得擅自引入 OpenVLA、其他 VLA，或面向多模型的 backend、adapter、policy registry 等抽象。
- 易碎物品安全机制是 **Open Research Question / TBD**。不得实现、命名或假定项目中已经存在 Safety Layer。

## 改动与文档同步

- 重大架构变更须先明确提出，并在实施时同步更新 `docs/architecture.md`。
- 项目阶段、已确认事实、研究问题或下一里程碑变化时，同步更新 `docs/status.md`。
- 未确认的接口、参数和技术方案必须标记为 TBD，不得写成项目事实。
- 用户主要负责架构与设计；队友主要负责 Isaac Sim/robot 环境实现。涉及双方边界的修改，应在文档中说明依赖与接口。

## Git 与产物

- 代码、配置和项目事实进入 GitHub。
- 数据集、视频、模型权重、checkpoint、运行日志及 `runs/` 等大文件或生成产物不得提交，应通过 `.gitignore` 排除并保存在本地或外部存储。
- 不提交密钥、令牌、机器专属路径或其他敏感信息。
