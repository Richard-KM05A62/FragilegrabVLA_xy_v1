# Run 的 U 盘安全归档

`export_run_to_usb.py` 将一个已经完成的 run 复制到已挂载 U 盘，逐文件比较大小和 SHA-256，验证通过后才永久删除本地完整 run。随后在原路径创建轻量占位目录，保留：

- `archived_run.json`：任务、结果、目标位置、文件数量、原大小和整树校验值；
- `previews/.../base_rgb.webm`：默认保留的小体积预览视频。

完整原始帧、timeline、MP4、场景和审计文件只保存在 U 盘。占位目录不是完整 episode，不能交给原 episode validator。

## U 盘准备

建议把 U 盘格式化为 exFAT 或 ext4，并设置唯一卷标，例如 `FRAGILE_DATA`。桌面系统自动挂载后，可以按卷标运行，不依赖具体 `/media/...` 路径：

```bash
python3 scripts/archive/export_run_to_usb.py \
  runs/fragilemanip_v01_qualification \
  --usb-label FRAGILE_DATA
```

也可以显式指定挂载点：

```bash
python3 scripts/archive/export_run_to_usb.py \
  runs/fragilemanip_v01_qualification \
  --usb-root /media/zhangbuhan/FRAGILE_DATA
```

脚本要求该路径本身是独立挂载点，避免 U 盘未挂载时误写本机磁盘。U 盘中的目标位置固定为：

```text
<U盘>/FragileGrabVLA/runs/<run目录名>/
```

目标已存在、空间不足、run 未完成、含符号链接、复制中断或校验失败时，脚本拒绝删除本地 run。归档成功后应正常弹出 U 盘，不要直接拔出。

如果连预览视频也不想留在本地，可以添加：

```bash
--no-local-preview
```
