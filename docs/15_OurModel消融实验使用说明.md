# OurModel 消融实验使用说明

## 1. 实现边界

消融实验采用新增模块实现，原始 `models/our/our_model.py` 未修改。

- 消融模型：`models/ourablation_model.py`
- 统一入口：`experiments/run_ablation.py`
- 自动化测试：`tests/test_our_ablation.py`
- 原始 OurModel：继续由 `--model our` 创建，行为不变
- 消融模型：由 `--model ourablation` 创建，只供消融入口使用

不指定消融变体时，`full` 变体调用原 OurModel 的原始 `forward`。自动化测试会加载同一组权重并检查两者 logits 逐元素一致。

## 2. 可用变体

| 变体 | 含义 |
|---|---|
| `full` | 完整 OurModel |
| `no_personality` | 人格投影槽位置零 |
| `no_vem` | 旁路视觉增强模块 VEM |
| `no_cfm` | 旁路完整双向反馈模块 CFM |
| `no_a2v` | 关闭音频到视觉反馈，保留视觉到音频反馈 |
| `no_v2a` | 关闭视觉到音频反馈，保留音频到视觉反馈 |
| `no_aux_focal` | 辅助 Focal Loss 权重置零，只保留主 CE |
| `audio_only` | 只执行音频编码器，其余融合槽位置零 |
| `visual_only` | 只执行视觉编码器及 VEM，其余融合槽位置零 |
| `personality_only` | 只执行人格投影，音频和视觉槽位置零 |

## 3. 输出隔离

默认输出根目录：

```text
experiments/results_ablation/<年份>/
```

每个变体拥有独立目录：

```text
experiments/results_ablation/2026/no_vem/
├── ablation_config.json
├── raw_results.csv
├── raw/
├── predictions/
├── confusion_matrix/
└── independent_test_results.csv
```

统一汇总文件：

- 五折结果：`ablation_raw_results.csv`
- 独立测试结果：`ablation_independent_test_results.csv`

两个汇总文件均新增 `Ablation` 列，用于区分变体。原八模型对比实验的 CSV、checkpoint 和日志不会被覆盖。

## 4. 2025 五折消融训练

在已经激活 `dachuangxiangmu` 环境后执行：

```powershell
python experiments\run_ablation.py --stage cv --variants "full,no_personality,no_vem,no_cfm,no_a2v,no_v2a,no_aux_focal,audio_only,visual_only,personality_only" --protocol legacy_bicfnet --dataset-year 2025 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2025" --audio-feature mfccs --video-feature densenet --personality-id-source subject_id --split-window 1s --folds 5 --seed 2024 --device cuda --results-dir "experiments\results_ablation\2025"
```

## 5. 2025 独立测试

必须在对应五折训练全部成功后执行：

```powershell
python experiments\run_ablation.py --stage independent-test --variants "full,no_personality,no_vem,no_cfm,no_a2v,no_v2a,no_aux_focal,audio_only,visual_only,personality_only" --protocol legacy_bicfnet --dataset-year 2025 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2025" --audio-feature mfccs --video-feature densenet --personality-id-source subject_id --split-window 1s --folds 5 --seed 2024 --device cuda --results-dir "experiments\results_ablation\2025"
```

## 6. 2026 五折消融训练

```powershell
python experiments\run_ablation.py --stage cv --variants "full,no_personality,no_vem,no_cfm,no_a2v,no_v2a,no_aux_focal,audio_only,visual_only,personality_only" --protocol legacy_bicfnet --dataset-year 2026 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2026" --audio-feature mfccs --video-feature densenet --personality-id-source subject_id --split-window 1s --folds 5 --seed 2024 --device cuda --results-dir "experiments\results_ablation\2026"
```

## 7. 2026 独立测试

```powershell
python experiments\run_ablation.py --stage independent-test --variants "full,no_personality,no_vem,no_cfm,no_a2v,no_v2a,no_aux_focal,audio_only,visual_only,personality_only" --protocol legacy_bicfnet --dataset-year 2026 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2026" --audio-feature mfccs --video-feature densenet --personality-id-source subject_id --split-window 1s --folds 5 --seed 2024 --device cuda --results-dir "experiments\results_ablation\2026"
```

## 8. 分批运行

可以只运行部分变体，例如先验证三个核心配置：

```powershell
python experiments\run_ablation.py --stage cv --variants "full,no_vem,no_cfm" --protocol legacy_bicfnet --dataset-year 2026 --cohort Elder --data-root "D:\Files\Works\CreationProject\test\2026" --personality-id-source subject_id --seed 2024 --device cuda --results-dir "experiments\results_ablation\2026"
```

同一变体、同一条件重新运行时，既有五折结果按 Run ID 与 Fold 更新；不同变体位于不同目录，不互相覆盖。

## 9. 论文使用边界

- 所有变体必须采用同一份 subject-level Fold；
- 所有变体必须按验证集 Macro-F1 保存最佳 checkpoint；
- 不得用历史冻结 checkpoint 作为 `full`，同时用新训练权重作为消融行；
- 不得根据独立测试集结果选择 seed 或训练轮次；
- 2025 历史混淆矩阵 `[[187,0],[14,26]]` 只能作为历史复现结果单列；
- 正式消融主表应报告 subject-level Macro-F1，并同时报告 Accuracy、Sensitivity、Specificity 和混淆矩阵。

## 10. 本轮验证范围

本轮只进行模型构造、前向传播、反向传播、CLI 和自动化测试，不自动启动完整 10 变体 × 5-Fold 长时间训练。
