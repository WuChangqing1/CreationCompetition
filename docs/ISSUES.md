# 问题记录

> 记录开发中遇到的 bug、坑、注意事项

---

## Known Issues

### K001 — Young Track2 test 视频特征为空
- **发现日期**: 2026-06-02
- **来源**: 官方基线 README
- **详情**: `MPDD-AVG2026-test/Young/Video/densenet`、`resnet`、`openface` 下文件为 0 字节空文件
- **影响**: Track2 A-V+P 和 A-V-G+P 在 test 阶段视频分支退化为零输入，评分会偏低
- **状态**: 等待组织方更新数据

### K002 — Young trainval OpenFace 部分无效
- **发现日期**: 2026-06-02
- **来源**: 官方基线 README
- **详情**: Young trainval 的 Video/openface 不是全量有效文件
- **影响**: 使用 OpenFace 特征时有效样本数减少
- **处理**: 训练时结合日志中的有效样本数判断

---

（后续问题在开发过程中自动追加）
