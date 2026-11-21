# Piper X H40 RTC runtime

本目录是项目根目录的最小运行子集，内部相对路径保持不变。

1. 将本目录内容复制到新机器的 `PIPER_PROJECT_ROOT`。
2. 将H40模型包解压到同一个项目根目录。
3. 按 `docs/environment/LINUX_INFERENCE_MIGRATION_GUIDE.md`配置环境和相机支持。
4. 按 `docs/inference/LINUX_RTC_INFERENCE_RUNBOOK.md`依次完成GPU、设备、单次只读和完整RTC只读验收。

本包不包含模型权重、虚拟环境、NVIDIA驱动、系统内核模块或完整Orbbec SDK。

