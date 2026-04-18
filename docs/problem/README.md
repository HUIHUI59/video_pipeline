# Known Problems & Decisions Log

本目录记录 pipeline 开发中遇到的**有决策价值的问题**——不是一次性的报错，而是"有多条路径、需要选择、日后要回溯"的事。

每个文件对应一个问题，格式：
- 现象
- 根因分析
- 可选方案（含已否决的）
- 当前选择 + 理由
- 待复测 / 时间点

## 目录

| 文件 | 主题 | 状态 |
|---|---|---|
| [01_stage5_output_truncation.md](./01_stage5_output_truncation.md) | Stage 5 VLM 输出 JSON 截断 | 已选两轮标注方案（E），待实现 |
