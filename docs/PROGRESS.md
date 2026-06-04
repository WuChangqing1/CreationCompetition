# 项目进度

> 最后更新：2026-06-03
> 本文件由 Agent 自动初始化

## 进行中
- [ ] 实现 5-fold 交叉验证（替代 9:1 单次划分）
- [ ] 实现 Elder+Young 联合训练（G+P 优先，二分类 175 样本）

## 待办
- [ ] 下载 MPDD-AVG 2026 test 集（trainval 已就位，test 空）
- [ ] 试 hybrid_attn 编码器替代 bilstm_mean
- [ ] 调优 A-V+P 超参：提高 hidden_dim/lr/epochs，匹配高维特征需求
- [ ] 跑 Track2 Young 基线（数据分布可能不同，超参更激进）
- [ ] 修复 train.py 缺失的高级特性：weighted_sampler, label_smoothing, cls/reg loss weights
- [ ] 特征降维：对 1000d 视频特征做 PCA 降至 128-256d
- [ ] 数据增强：对音频片段做时间偏移/加噪
- [ ] 修复 Elder 5 人不完整样本（仅 2 片段）
- [ ] 模型架构改进实验
- [ ] 特征融合策略优化
- [ ] 生成 CodaBench 提交文件

## 已完成
- [x] [2026-06-04] 数据集全面分析：17,231 个文件，9 大章节（概览、目录、标签、特征维度、数据质量、划分、跨数据集差异、诊断、改进建议），写入 数据集分析.md
- [x] [2026-06-03] run_baseline.py 修复：移除 train.py 不接受的 extra CLI 参数（--selection_metric, --cls_loss_weight 等）
- [x] [2026-06-03] 基线训练首轮完成：G+P binary (F1=0.75, kappa=0.50) + ternary (失败), A-V+P binary (F1=0.50) + ternary (F1=0.36)
- [x] [2026-06-03] 诊断结论：① train.py 缺失 shell 脚本引用的高级特性 ② 78 样本 vs 2100 维特征导致维度灾难 ③ 三元分类当前数据规模不可行
- [x] [2026-06-02] 项目初始化：docs/ 目录及所有记忆文件
- [x] [2026-06-02] 基线代码入库（MPDD-AVG-2026 + MPDD 2025）
- [x] [2026-06-02] 创建 GitHub 私有仓库 CreationCompetition
- [x] [2026-06-02] 初始化项目架构文档
- [x] [2026-06-02] 创建 conda 环境 creationcompetition（Python 3.10.20, CUDA 12.9, GPU 可用）
- [x] [2026-06-02] 安装全部依赖：PyTorch 2.7.1+cu128 + 14 个 ML 包（阿里云镜像加速）
- [x] [2026-06-02] 全项目文件盘点：178 个文件清单
- [x] [2026-06-02] 数据集盘点：Elder 87人 + Young 88人，全部特征就位，发现 5 个数据问题 (K003-K007)
- [x] [2026-06-02] 创建 requirements.txt（14 个 pip 依赖）
- [x] [2026-06-02] 创建 smoke_test.py（4 阶段冒烟测试全部通过）
- [x] [2026-06-02] 创建 run_baseline.py（统一批量训练脚本，替代 12 个 shell 脚本）
