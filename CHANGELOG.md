# Changelog

## 1.2.4

- 结构模型从 SongFormer 恢复为 All-In-One 1.1.0。
- 删除 SongFormer worker、独立环境安装器、CUDA 修复器和相关设置。
- All-In-One 直接在主进程的后台线程中运行，不启动额外 Python/Conda 进程。
- 新增独立的 All-In-One 设备设置，默认 CPU，避免占满 RTX 5070 显存。
- Beat This! 与 MuQ 仍可单独使用 CUDA。
- All-In-One、MuQ、Beat This! 和滑动预加载加入互斥执行保护。
- 保留 GUI 分析进度条与 PowerShell 实时日志。
- 支持 harmonix-all 和 harmonix-fold0 至 harmonix-fold7。
- All-In-One 缓存升级为 v3。
- 37 项自动测试通过。
