# MPDD 2025 Multi-model Experiment Framework Implementation Plan

> 历史计划说明：本文记录 2026-09-06 的原始实施过程。自 2026-09-07 起，Task 10 所述 DepMamba 与 Proposed 已按用户决定退出正式实验范围，相关占位实现与测试已移除；当前口径以 `docs/CURRENT_STATUS.md` 为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 MPDD-main 增量改造成以 `models.create_model()` 为 PyTorch 统一入口、以 `experiments/model_registry.py` 为整体调度入口的可复现多模型实验框架。

**Architecture:** PyTorch adapter 统一遵循 BaseModel 输入/输出约定，传统模型留在 experiments/classical。Subject-level folds、训练、评估、效率和结果保存各自为独立模块，train/test 保留旧命令兼容并改用统一模型入口。

**Tech Stack:** Python 3.10, PyTorch, NumPy, pandas, scikit-learn, optional XGBoost, optional matplotlib, optional mamba_ssm.

**Spec:** `docs/superpowers/specs/2026-09-06-mpdd-multimodel-experiment-framework-design.md`

## Global Constraints

- 所有 Python 执行只允许 `conda run -n dachuangxiangmu python ...`。
- 不创建 Conda 环境，不修改 base，不调用裸 `pip`，不自动安装第三方依赖。
- 不删除或破坏 Track2、三分类、五分类、scripts、checkpoints、logs 和旧 checkpoint。
- 不运行完整的所有模型 5-Fold 长时训练；只运行 import、单 batch、合成数据和 tiny smoke。
- PyTorch 模型唯一创建入口是 `models.create_model()`；SVM/XGBoost 不继承 BaseModel。
- 不伪造数据、指标或 PASS；真实数据不可见时 Dataset 状态必须是 SKIPPED。
- 新脚本必须有 `if __name__ == "__main__":`。
- 当前目录不是 Git 仓库，因此所有任务的提交步骤改为更新 CURRENT_STATUS 与检查文件清单。

---

### Task 1: 持久上下文与环境文档

**Files:** Create `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/CURRENT_STATUS.md`, `docs/00_项目结构说明.md`, `docs/07_环境与依赖说明.md`, `docs/CHANGELOG_EXPERIMENTS.md`.

**Interfaces:** Consumes 已批准设计、源码勘察与环境输出；produces 后续任务使用的稳定规则、技术事实和动态状态。

- [ ] **Step 1: 写短而稳定的 AGENTS.md**

包含唯一环境、两层 registry、公平性、质量约束、文档索引和完成标准，不复制长篇技术说明。

- [ ] **Step 2: 写源码事实与阶段状态**

PROJECT_CONTEXT 记录数据键、shape、OurModel、BaseModel、train/test 和 checkpoint；CURRENT_STATUS 初始化阶段表，真实 Dataset 标为 SKIPPED。

- [ ] **Step 3: 写 00、07 与 changelog**

07 同时记录直接 activate 失败与强制 `conda run` 成功的路径、版本、CUDA/GPU、缺失依赖。

- [ ] **Step 4: 验证文件编码和必需章节**

Run: `conda run -n dachuangxiangmu python -c "from pathlib import Path; files=['AGENTS.md','docs/PROJECT_CONTEXT.md','docs/CURRENT_STATUS.md','docs/00_项目结构说明.md','docs/07_环境与依赖说明.md','docs/CHANGELOG_EXPERIMENTS.md']; [Path(p).read_text(encoding='utf-8') for p in files]; print('context docs OK')"`

Expected: `context docs OK`.

### Task 2: Registry 与 OurModel 兼容

**Files:** Create `tests/test_model_registry.py`; modify `models/__init__.py`, `train.py`, `test.py`.

**Interfaces:** Consumes `find_model_using_name(name: str) -> type[BaseModel]`; produces 明确异常的动态发现、`--model` 参数、统一创建和 `unpack_checkpoint(payload, fallback_model)`。

- [ ] **Step 1: 写 registry 与 checkpoint 失败测试**

```python
def test_our_model_is_created_through_registry(make_opt):
    model = create_model(make_opt(model="our", isTrain=False))
    assert type(model).__name__ == "ourModel"

def test_checkpoint_payload_accepts_legacy_and_metadata():
    assert unpack_checkpoint({"w": 1}, "our")[0] == "our"
    assert unpack_checkpoint({"model_name": "mlp", "model_state_dict": {"w": 1}}, "our")[0] == "mlp"
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_model_registry -v`

Expected: FAIL because checkpoint helper/strict error behavior is missing.

- [ ] **Step 3: 实现明确 registry 错误和统一创建**

精确路由 `our`；找不到类时抛 `ImportError`，不 `exit(0)`。train/test 增加默认 `--model our` 并调用 `create_model(opt)`；test 同时识别新旧 checkpoint。

- [ ] **Step 4: 验证 GREEN**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_model_registry -v`

Expected: PASS. 更新 CURRENT_STATUS 和 changelog，记录三处统一入口证据。

### Task 3: 公共 PyTorch adapter、MLP、BiLSTM、LightWeightTrans

**Files:** Create `models/av_model.py`, `models/mlp_model.py`, `models/bilstm_model.py`, `models/lightweighttrans_model.py`, `tests/test_simple_models.py`; modify `models/networks/lstm.py`.

**Interfaces:** Produces `masked_mean(x, mask)`, `AVClassificationModel` 和三个输出 `[B,C]` logits 的完整模型。

- [ ] **Step 1: 写参数化失败测试**

```python
def test_simple_models_forward_binary_logits(make_opt, sample_batch):
    for name in ("mlp", "bilstm", "lightweighttrans"):
        model = create_model(make_opt(model=name, isTrain=True))
        model.set_input(sample_batch)
        model.forward()
        assert tuple(model.emo_logits.shape) == (2, 2)
        assert torch.allclose(model.emo_pred.sum(1), torch.ones(2), atol=1e-5)
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_simple_models -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: 实现公共 adapter 与 MLP，验证 MLP**

公共类负责 device-safe set_input、CE、Adam/AdamW、softmax 和 masked mean。MLP 用 pooled A/V 与可选 P。

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_simple_models.SimpleModelsTest.test_mlp_forward -v`

Expected: PASS.

- [ ] **Step 4: 实现并验证 BiLSTM**

复用 `LSTMEncoder(..., bidirectional=True)`，修正双向输出维度，再投影和池化。

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_simple_models.SimpleModelsTest.test_bilstm_forward -v`

Expected: PASS.

- [ ] **Step 5: 实现并验证 LightWeightTrans adapter**

adapter 在 batch-first 与 sequence-first 之间转换后池化。

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_simple_models -v`

Expected: all PASS. 更新状态文档。

### Task 4: Classical baseline 与 Experiment Registry

**Files:** Create `experiments/__init__.py`, `experiments/classical/__init__.py`, `experiments/classical/common.py`, `experiments/classical/svm_baseline.py`, `experiments/classical/xgboost_baseline.py`, `experiments/model_registry.py`, `tests/test_experiment_registry.py`.

**Interfaces:** Produces `pool_multimodal_features(A,V,P,use_personality=True)`, `get_model_kind(name)`, `create_experiment_model(name,opt)`.

- [ ] **Step 1: 写 pooling、SVM pipeline 和缺失 XGBoost 测试**

```python
def test_pooling_uses_temporal_mean():
    out = pool_multimodal_features(np.array([[[1.,3.],[3.,5.]]]), np.array([[[2.],[4.]]]), np.array([[9.]]))
    np.testing.assert_allclose(out, [[2.,4.,3.,9.]])

def test_svm_contains_scaler_before_classifier():
    assert list(create_svm().named_steps) == ["scaler", "classifier"]
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_experiment_registry -v`

Expected: FAIL on missing experiments modules.

- [ ] **Step 3: 实现 pooling、SVM、XGBoost 延迟导入与两层 registry**

XGBoost 只有被选择时 import；缺失时抛含安装命令的 `OptionalDependencyError`。

- [ ] **Step 4: 验证 GREEN**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_experiment_registry -v`

Expected: SVM behavior and XGBoost-unavailable behavior PASS.

### Task 5: LMF 与 MulT

**Files:** Create `models/networks/lmf.py`, `models/lmf_model.py`, `models/networks/mult/{__init__,attention,transformer,network}.py`, `models/mult_model.py`, `tests/test_fusion_models.py`.

**Interfaces:** Produces `LowRankFusion.forward(a,v,p)->Tensor[B,F]` and `MulTNetwork.forward(a,v,a_mask,v_mask)->Tensor[B,F]`.

- [ ] **Step 1: 写 LMF/MulT 失败测试**

```python
def test_fusion_models_forward_binary_logits(make_opt, sample_batch):
    for name in ("lmf", "mult"):
        model = create_model(make_opt(model=name, isTrain=True))
        model.set_input(sample_batch)
        model.forward()
        assert tuple(model.emo_logits.shape) == (2, 2)
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_fusion_models -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: 实现 LMF 并验证**

使用每模态增广常数 1、rank 维低秩乘积和输出投影，参数全部注册为 `nn.Parameter`。

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_fusion_models.FusionModelsTest.test_lmf_forward -v`

Expected: PASS.

- [ ] **Step 4: 实现 MulT 并验证**

音频查询视觉、视觉查询音频，各自经过残差 cross-attention 和 encoder，掩码池化后拼接；P 在 adapter 末端融合。

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_fusion_models -v`

Expected: all PASS. 更新状态文档。

### Task 6: Subject-level Fold

**Files:** Create `experiments/data_utils.py`, `experiments/create_splits.py`, `tests/test_subject_splits.py`.

**Interfaces:** Produces `extract_subject_id(entry)->str`, `build_subject_labels(entries,label_key)->dict[str,int]`, `create_subject_folds(entries,label_key,folds,seed)->list[dict]`, `validate_no_subject_leakage(train_ids,val_ids)`.

- [ ] **Step 1: 写泄漏、复用与分层失败测试**

```python
def test_leakage_is_rejected():
    with self.assertRaisesRegex(ValueError, "Subject leakage detected"):
        validate_no_subject_leakage(["1","2"], ["2","3"])

def test_each_subject_appears_in_one_validation_fold():
    folds = create_subject_folds(make_balanced_entries(), "bin_category", 5, 3407)
    ids = [sid for fold in folds for sid in fold["val_ids"]]
    self.assertEqual(len(ids), len(set(ids)))
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_subject_splits -v`

Expected: FAIL on missing module.

- [ ] **Step 3: 实现 aggregation、StratifiedKFold、JSON 原子写入和默认复用**

类别少于 folds 时抛包含最小类别受试者数量的错误。

- [ ] **Step 4: 验证 GREEN**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_subject_splits -v`

Expected: PASS.

### Task 7: Evaluator、预测与 Confusion Matrix

**Files:** Create `experiments/evaluator.py`, `tests/test_evaluator.py`.

**Interfaces:** Produces `evaluate_predictions(y_true,y_pred,y_prob)->dict`, `save_predictions(...)`, `save_confusion_matrix(...)`.

- [ ] **Step 1: 写手算指标与单类 AUC 测试**

```python
def test_binary_metrics_match_hand_checked_values():
    result = evaluate_predictions([0,0,1,1], [0,1,1,1], [[.9,.1],[.4,.6],[.2,.8],[.1,.9]])
    self.assertEqual(result["Accuracy"], 0.75)
    self.assertEqual(result["Positive_Recall"], 1.0)
    self.assertEqual(result["Specificity"], 0.5)

def test_single_class_auc_is_na():
    self.assertEqual(evaluate_predictions([0,0], [0,0], [[1,0],[1,0]])["ROC_AUC"], "N/A")
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_evaluator -v`

Expected: FAIL on missing evaluator.

- [ ] **Step 3: 实现指标、CSV 与可选 matplotlib 绘图**

matplotlib 缺失时保存 confusion matrix CSV 并返回 SKIPPED reason，不伪造 PNG。

- [ ] **Step 4: 验证 GREEN**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_evaluator -v`

Expected: PASS.

### Task 8: 配置、Checkpoint、统一 CV 与效率

**Files:** Create `experiments/configs/*.json`, `experiments/checkpointing.py`, `experiments/efficiency.py`, `experiments/run_model_cv.py`, `tests/test_cv_support.py`.

**Interfaces:** Produces `set_random_seed(seed)`, `build_checkpoint_payload(...)`, `measure_torch_efficiency(...)`, `run_fold(args,fold_data)->dict`.

- [ ] **Step 1: 写 seed、checkpoint 元数据与模型创建路径失败测试**

```python
def test_checkpoint_contains_reproducibility_metadata():
    payload = build_checkpoint_payload("mlp", {"w":1}, {"lr":1e-3}, {"audio":"mfccs"}, 2, 3407)
    self.assertEqual(set(payload), {"model_name","model_state_dict","config","feature_config","fold","seed"})
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_cv_support -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: 实现配置、seed、checkpoint 和效率统计**

效率默认 warmup 10、measure 50；CPU peak VRAM 返回 N/A。

- [ ] **Step 4: 实现 run_model_cv**

包含参数解析、fold 复用、classical/torch 分支、结果追加和 OOF 汇总；`--tiny` 限制 epoch/batch，不在 import 时执行。

- [ ] **Step 5: 验证 GREEN 与帮助入口**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_cv_support -v`

Run: `conda run -n dachuangxiangmu python experiments/run_model_cv.py --help`

Expected: tests PASS and help exits 0.

### Task 9: run_all 与聚合结果

**Files:** Create `experiments/run_all.py`, `experiments/aggregate_results.py`, `tests/test_orchestration.py`.

**Interfaces:** Produces `run_models(model_names,runner,report_path)->list[dict]`, `aggregate_results(rows)->list[dict]`.

- [ ] **Step 1: 写失败隔离和 mean±std 失败测试**

```python
def test_failed_model_does_not_stop_following_models():
    report = run_models(["bad","good"], fake_runner, report_path)
    self.assertEqual([r["status"] for r in report], ["FAILED","PASS"])

def test_aggregate_does_not_invent_missing_values():
    summary = aggregate_results([{"Model":"mlp","Accuracy":0.5,"Status":"PASS"}])
    self.assertEqual(summary[0]["Accuracy"], "0.5000 ± 0.0000")
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_orchestration -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: 实现子进程调度、状态映射、run_report 和聚合**

子进程使用 `sys.executable` 保持 dachuangxiangmu；聚合只读取实际 raw_results。

- [ ] **Step 4: 验证 GREEN 与帮助入口**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_orchestration -v`

Run: `conda run -n dachuangxiangmu python experiments/run_all.py --help`

Expected: PASS and help exits 0.

### Task 10: DepMamba 与 Proposed 安全状态

**Files:** Create `models/depmamba_model.py`, `models/proposed_model.py`, `models/networks/depmamba/__init__.py`, `tests/test_optional_models.py`.

**Interfaces:** Produces registry 可发现类；DepMamba 缺依赖时报 `OptionalDependencyError`；Proposed 实例化时报 `NotImplementedError`。

- [ ] **Step 1: 写延迟失败测试**

```python
def test_optional_import_does_not_break_other_models():
    self.assertIsNotNone(find_model_using_name("depmamba"))
    self.assertIsNotNone(find_model_using_name("mlp"))

def test_proposed_is_explicitly_unimplemented(make_opt):
    with self.assertRaisesRegex(NotImplementedError, "has not been implemented"):
        create_model(make_opt(model="proposed"))
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_optional_models -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: 实现延迟依赖检查与明确状态**

不在模块顶层 import mamba_ssm；只在 DepMamba 构造时检查并报缺失包列表。

- [ ] **Step 4: 验证 GREEN**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_optional_models -v`

Expected: PASS.

### Task 11: 分阶段 Smoke Test

**Files:** Create `experiments/smoke_test.py`, `tests/test_smoke_contract.py`.

**Interfaces:** Produces `run_smoke(data_root=None,device="cpu")->list[SmokeResult]`; status 只能为 PASS、FAIL、SKIPPED。

- [ ] **Step 1: 写 smoke 状态和关键项目失败测试**

```python
def test_smoke_reports_required_components():
    names = {r.name for r in run_smoke(data_root=None, device="cpu")}
    self.assertTrue({"Environment","Dataset","our","mlp","bilstm","lightweighttrans","lmf","mult","Evaluator","Subject Split"} <= names)
```

- [ ] **Step 2: 验证 RED**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_smoke_contract -v`

Expected: FAIL on missing smoke runner.

- [ ] **Step 3: 实现环境、真实/合成 Dataset、registry、forward、evaluator 和 split smoke**

未提供真实 data root 时 Dataset=SKIPPED；合成 batch 单独标明，不替代真实 Dataset PASS。

- [ ] **Step 4: 验证 GREEN 并运行正式 smoke**

Run: `conda run -n dachuangxiangmu python -m unittest tests.test_smoke_contract -v`

Run: `conda run -n dachuangxiangmu python experiments/smoke_test.py`

Expected: 支持模型实际 PASS；缺失真实数据/xgboost/DepMamba/matplotlib 如实 SKIPPED。

### Task 12: 完整中文文档与最终验收

**Files:** Create `docs/01_多模型框架改造说明.md` through `docs/06_常见错误与解决方案.md`; modify `requirements.txt`, PROJECT_CONTEXT, CURRENT_STATUS, CHANGELOG.

**Interfaces:** Consumes 实际代码和新鲜测试输出；produces 面向本科生的运行、模型、指标、结果和排错手册。

- [ ] **Step 1: 写 01 与真实 OurModel Mermaid**

图中包含 LSTM A/V、ProjA/V、VEM、CFM、masked pooling、ProjP、EmoC/EmoCF。

- [ ] **Step 2: 写 02 至 06**

逐模型覆盖是什么、原因、输入、结构、输出、优缺点、位置、来源和 MPDD 适配；教程给完整 PowerShell 命令；错误统一为现象/原因/检查/解决。

- [ ] **Step 3: 更新 requirements 分组**

保留原内容并分 Core、Classical baseline、Results visualization、Optional DepMamba；不固定未经验证的 Windows wheel。

- [ ] **Step 4: 运行完整测试**

Run: `conda run -n dachuangxiangmu python -m unittest discover -s tests -v`

Expected: 0 failures and 0 errors.

- [ ] **Step 5: 运行关键 import 与正式 smoke**

Run: `conda run -n dachuangxiangmu python -c "import models; print('models OK')"`

Run: `conda run -n dachuangxiangmu python -c "from experiments import evaluator; print('evaluator OK')"`

Run: `conda run -n dachuangxiangmu python experiments/smoke_test.py`

Expected: imports exit 0; smoke reflects actual PASS/SKIPPED/FAIL.

- [ ] **Step 6: 打印关键目录与核对禁止项**

Run separately: `tree models /F`, `tree experiments /F`, `tree docs /F`.

Run: `rg -n "from models\.our\.our_model import ourModel|model = ourModel" train.py test.py experiments models`

Expected: tree 包含目标文件；硬编码模型查询无命中。

- [ ] **Step 7: 回读计划与规范逐项验收**

将实际失败、缺依赖和未运行项写入 CURRENT_STATUS、CHANGELOG 和最终汇报，不把部分验证表述成全部通过。
