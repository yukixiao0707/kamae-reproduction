# C 独立验证入口

已完成 A 原始复跑、128 行原校验、1,000,209 行全量校验和 Bucketize 扩展。全量四个输出均零误差；扩展 32 个案例存在 4 个反例。完整结论和表格见 [实验章节初稿](handoff/实验章节初稿.md)。

按 A 文档详细程度补写的 [完整实验记录](实验记录.md) 包含准备、参数、具体操作、报错处理、运行时间、全量统计、32 个扩展案例和原始证据链接。章节初稿供报告整合，完整记录用于复核过程。

## 本机逐步运行

在 PowerShell 中执行（每次从项目根目录开始）：

```powershell
cd 'D:\a-learn\研究生\课程\大数据处理\kamae-reproduction'
& .\reproduction\c_validation\run_c.ps1 -Step diagnose
& .\reproduction\c_validation\run_c.ps1 -Step baseline
& .\reproduction\c_validation\run_c.ps1 -Step original-check
& .\reproduction\c_validation\run_c.ps1 -Step full-check
& .\reproduction\c_validation\run_c.ps1 -Step extension
& .\reproduction\c_validation\run_c.ps1 -Step probe
& .\reproduction\c_validation\run_c.ps1 -Step plot
& .\.venv-c310\Scripts\python.exe .\reproduction\c_validation\finalize.py
```

baseline 会重新拟合并覆盖本机 reproduction/outputs 中的生成产物。它们是运行结果，不是复跑前必须向 A 索取的数据。原始输入为 reproduction/data/ml-1m 下的三个 dat 文件。

## 文件用途

- ACCEPTANCE.md：实验运行前确定的验收标准。
- validate_full.py：分批进行全量精确比对，CSV/JSON 同时记录形状、dtype 与误差。
- extension_bucketize.py：C 的独立扩展及模型保存/重新加载比较。
- probe_bucketize.py：直接运行底层 TensorFlow 算子，定位分桶差异。
- handoff/：交给 B 的章节初稿、表格、图和扩展模型。
- results/：本机完整日志、每步时间和退出码；被 Git 忽略。
- assets_manifest.json、windows_native_manifest.json：输入和运行组件来源、哈希。

## 本机环境设置

使用项目 .venv-c310；Python、Spark、Java 和 Hadoop 辅助组件另存于用户 Documents/Codex/work 下，具体路径由 run_c.ps1 设置。它是这台电脑已配置环境的运行入口，迁移到其他电脑需修改相应路径，不是通用的一键安装器。

Spark 使用 PyPI 4.2.1 源包附带的原始 Python 包和 jar，未更改其中计算逻辑。受限 Windows 环境不能修改 ACL，因此 runtime_compat 只让本实验目录的临时文件和输出继承已有权限；InheritedAclLocalFileSystem.java 编译为 kc_runtime_fs.jar，放在本机 Spark jars 目录。若在常规终端自行配置 Hadoop 原生权限工具，可以删除该自定义 fs.file.impl 配置后使用标准文件系统。不要将这项兼容设置描述为论文算法的修改。

基线结束时日志中的 `ERROR: Access denied` 来自 Windows 子进程清理命令；基线计算已完成，退出码为 0，随后生成文件已被独立读取并完成全量校验。早期环境失败日志也保留，未计入通过结果。

## 检查结果

C 新增代码完成 Black/isort、关键语法和类型注解 lint、验证器的 3 项检查。图表保留源数据、源码、PDF 字体与重叠检查记录。没有修改 A 脚本或 Kamae 库源码，也没有运行上游库的全部测试矩阵。
