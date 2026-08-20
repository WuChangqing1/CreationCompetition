# 多 Seed 测试方案（完整文档）

> 本文档是 MPDD-AVG 2026 项目「多 seed 测试」的完整操作手册：
> 覆盖测试目的、测试矩阵、运行方式、常见报错与解决、结果汇总方法，以及当前全部进度。
> 最后更新：2026-08-20

---

## 0. 一句话说明

小样本（Elder 87 人 / Young 88 人）下，5-fold CV 单次（seed=42）的方差很大（Macro-F1 的 std 常达 0.1），单 seed 结论不可靠。因此用**多个随机种子**分别跑完整 5-fold CV，再对每个「方法 × 配置」取跨 seed 的**均值 ± 标准差**，得到更稳健的结论。

---

## 1. 测试矩阵

### 1.1 三个维度

| 维度 | 取值 | 说明 |
|---|---|---|
| 方法 | baseline(cross_fusion) / CMG-VS(cvae) / PTMFIM / HOPE / MSF-ATS(reliability) / P3HF(hypergraph) | 6 个 |
| 配置 | Track1+Track2 × `A-V+P` / `A-V-G+P` / `G+P` | 6 个（cvae 的 G+P 为 N/A） |
| 种子 | 42 / 2024 / 2025 | 3 个 |

- **总运行次数**：5 个方法 × 6 配置 + 1 个方法(cvae) × 4 配置 = 34 次/seed；× 3 seed = **102 次完整 5-fold CV**。
- 每次运行都是**完整 5-fold CV**（不是 quick 模式），全部 `--cls_only` 口径（关闭 PHQ 回归头，与之前对比实验一致）。

### 1.2 固定不变的设置

- 骨干：DepFormer 时序编码 + BCT 音视频协同。
- 任务：binary 二分类；folds=5。
- 特征：audio=mfcc、video=densenet。
- 环境：`dachuangxiangmu`（Python 3.10.20 / PyTorch 2.14.0.dev20260717+cu130 / RTX 5070）。
- 每个方法×配置×seed 的 `--seed` 不同，其余参数完全一致，保证只有「随机种子」这一个变量。

---

## 2. 运行方式（怎么进行多 seed 测试）

### 2.1 环境前置（Windows 必做）

```powershell
# 1) opensmile 依赖 audresample 需要这个环境变量，否则报找不到 DLL
$env:PROCESSOR_ARCHITECTURE = 'AMD64'

# 2) 直接调用环境内 python（避免 conda run 的 GBK 编码崩溃）
$py = 'D:\App\Business\Coding\Python\Miniconda\envs\dachuangxiangmu\python.exe'

# 3) 切到项目根目录
Set-Location 'D:\Files\Works\CreationProject'
```

### 2.2 单次运行的命令模板

```powershell
# baseline（cross_fusion）
& $py experiments/train_cv.py --track {Track1|Track2} --task binary --subtrack {A-V+P|A-V-G+P|G+P} --folds 5 --cls_only --seed {42|2024|2025} --num_workers 0

# CMG-VS（cvae，只用于 A-V+P / A-V-G+P，G+P 为 N/A）
& $py experiments/train_cv.py --track {Track1|Track2} --task binary --subtrack {A-V+P|A-V-G+P} --folds 5 --use_cvae --seed {seed} --num_workers 0

# 其余 4 个融合模块
& $py experiments/train_cv.py --track {Track1|Track2} --task binary --subtrack {S} --folds 5 --cls_only --fusion_type {ptmfim|hope|reliability|hypergraph} --seed {seed} --num_workers 0
```

参数说明：
- `--num_workers 0`：在**沙箱/受限环境**必须用 0（否则 DataLoader 多进程会因命名管道被拦截报 `WinError 5`）；**你自己本机**可去掉这个参数（默认 2，更快）。
- `--cls_only`：关闭 PHQ 回归头，与历史对比口径一致。
- 结果自动保存到 `experiments/cv_logs/{Track}/{Subtrack}/binary/cv_result_5fold_{时间戳}.json`。

### 2.3 结果 JSON 字段说明

每个 JSON 里的关键字段：

| 字段 | 含义 |
|---|---|
| `track` / `subtrack` / `task` | 配置 |
| `seed` | 随机种子 |
| `variant` | 方法（cvae / cross_fusion / ptmfim / hope / reliability / hypergraph） |
| `config` | 完整超参（含 `fusion_type`、`use_cvae` 等） |
| `cv_f1_mean` / `cv_f1_std` | 该 seed 下 5 折 Macro-F1 的均值/标准差 |
| `cv_acc_mean` / `cv_kappa_mean` | 该 seed 下 5 折 Acc / Kappa 均值 |
| `elapsed_sec` | 本次耗时（秒） |

### 2.4 并行编排策略

- 每个「方法×配置×seed」是独立进程，可并行。
- 但注意**显存**：每个 Python 进程会占一个 CUDA context（约几百 MB~1GB），8GB 显存建议**同时 4~8 个进程**，不要一次 68 个全开（会 OOM）。
- 推荐把「慢配置」（A-V-G+P，batch=2，约 7 分钟/次）和「快配置」（G+P 约 7s，A-V+P 约 87s）分组，慢的先进场并行跑。
- 沙箱环境下**不要用 `Start-Process` 派生子进程**（会被杀，日志为空）；用 harness 的「后台任务」或顺序 `&` 循环。

### 2.5 断点续跑（防丢）

1. 每次运行结束后，`experiments/cv_logs/...` 会留下带时间戳的 JSON。
2. 通过「是否存在某个 (track, subtrack, variant, seed) 的 JSON」判断是否已完成。
3. 缺哪个就单独补跑哪个，命令同上（幂等，重复跑只是覆盖结果）。

---

## 3. 常见报错与解决

### E1 — `PermissionError: [WinError 5] 拒绝访问`（DataLoader 多进程）
- **现象**：训练启动时报 `multiprocessing ... connection.Pipe` 或 `CreateFile` 拒绝访问。
- **原因**：受限/沙箱环境禁止创建命名管道，`num_workers>0` 的多进程加载会失败。
- **解决**：加 `--num_workers 0`。本机正常环境用默认 2 即可。

### E2 — opensmile / audresample 找不到 DLL
- **现象**：`import opensmile` 报找不到 `audresample\core\bin\win_\audresample.dll`。
- **原因**：当前进程 `platform.machine()` 返回空串，拼错平台目录。
- **解决**：先 `$env:PROCESSOR_ARCHITECTURE = 'AMD64'` 再运行。

### E3 — `conda run` 触发 `UnicodeEncodeError`（GBK）
- **现象**：`conda run -n dachuangxiangmu python ...` 崩溃，报 GBK 编码错误。
- **解决**：直接用环境内解释器 `D:\...\envs\dachuangxiangmu\python.exe`。

### E4 — CUDA 不可用但没报错（静默回退 CPU）
- **现象**：训练能跑但很慢、CPU 满、日志显示 `Device: cpu`。
- **解决**：已改为**硬报错**。现在 `train_cv.py` / `train.py` / `test.py` 在 `--device cuda` 且 CUDA 不可用时直接 `raise RuntimeError`。若真报错，检查 PyTorch 是否为 `+cu130` 版本、显卡驱动是否正常。

### E5 — CPU 满负荷、GPU 空转（数据管线瓶颈）
- **现象**：GPU 利用率很低、CPU 100%，训练极慢（体感像 CPU 训练）。
- **原因**：每次取样本都重复 `np.load + 归一化 + 插值`。
- **解决**：已在 `dataset.py` 加 `_FEATURE_CACHE` / `_cached_resize` 预缓存。实测 A-V+P 完整 5-fold 从 1888s 降到 **87s（约 22 倍加速）**，结果逐位一致。三个入口脚本共享此缓存。

### E6 — 显存不足 (OOM)
- **现象**：`CUDA out of memory`。
- **解决**：减少并行进程数（8GB 显存建议 ≤8 个）；或把 `A-V-G+P` 的 `batch_size` 从 2 提到 8 会更快但会改变该配置的历史口径（需统一重跑基线）。

### E7 — `FileNotFoundError`（train.py 路径解析）
- **现象**：`train.py` 报找不到 `...\Elder\split_labels_train.csv`。
- **原因**：`train.py` 相对路径基于 `test/MPDD-AVG-2026-main`（PROJECT_ROOT），不是项目根。
- **解决**：用**绝对路径**，或相对 PROJECT_ROOT 的 `../../Elder`（在项目根运行时）。

### E8 — checkpoint 互相覆盖
- **现象**：不同方法/seed 的结果存到同一个 `foldN_best.pth`。
- **解决**：已按 `variant`（cvae/cross_fusion/ptmfim/hope/reliability/hypergraph）分目录；但**不同 seed 仍共用同目录**，checkpoint 会被后跑的覆盖（**结果 JSON 不受影响**，汇总只看 JSON）。

### E9 — 会话中断导致后台任务被杀
- **现象**：后台任务只跑到 fold1/fold2 就没了、无结果 JSON。
- **解决**：重跑缺失的 (track, subtrack, variant, seed)；用「结果 JSON 是否存在」判断是否完成，幂等补跑。

### E10 — 训练崩在 `reg_out=None`（已修复）
- 历史问题：`use_cvae=False` 且 `use_regression_head=False` 时模型返回 `(logits, None)`，训练循环对 `None` 算 MSE 崩溃。已修复（按 `use_regression_head` 分支）。

---

## 4. 结果汇总方法

### 4.1 聚合逻辑

对每个「方法 × 配置」：
1. 找到该 (track, subtrack, variant) 下所有 seed（42/2024/2025）的结果 JSON。
2. 取每个 seed 的 `cv_f1_mean`（以及 `cv_acc_mean` / `cv_kappa_mean`）。
3. 计算**跨 seed 均值** 和 **跨 seed 标准差**：
   - `F1 = mean([f1_42, f1_2024, f1_2025])`
   - `std = std([f1_42, f1_2024, f1_2025])`

注意区分两个「std」：
- `cv_f1_std` = 单个 seed 内 5 折之间的标准差（折间波动）。
- **跨 seed std** = 不同 seed 之间的标准差（我们汇总关心的，反映「换个种子结果稳不稳」）。

### 4.2 汇总文档格式

最终写入 `docs/PAPER_COMPARISON_MULTISEED.md`，一张主表：

```
| 方法 | Track1 A-V+P | Track1 A-V-G+P | Track1 G+P | Track2 A-V+P | Track2 A-V-G+P | Track2 G+P |
```

每格填 `F1均值±跨seed std`（也可附 Acc / Kappa 详情表）。

---

## 5. 当前进度（截至 2026-08-20）

### 5.1 已完成：seed=42 单 seed 全部结果

（完整数字见 `docs/PAPER_COMPARISON.md`，下表为 Macro-F1 均值±折间std）

| 方法 | Track1 A-V+P | Track1 A-V-G+P | Track1 G+P | Track2 A-V+P | Track2 A-V-G+P | Track2 G+P |
|---|---|---|---|---|---|---|
| baseline (cross_fusion) | 0.6480 | 0.5297 | 0.4041 | 0.5462 | 0.5526 | 0.6809 |
| CMG-VS (cvae) | 0.6258 | 0.6479 | N/A | 0.4986 | 0.4994 | N/A |
| PTMFIM | 0.5749 | 0.5216 | 0.4041 | 0.5662 | 0.5461 | 0.6211 |
| HOPE | 0.5818 | 0.4679 | 0.4041 | 0.5035 | 0.5361 | 0.6254 |
| MSF-ATS (reliability) | 0.5239 | 0.5991 | 0.4441 | 0.5634 | 0.5572 | 0.6659 |
| P3HF (hypergraph) | 0.5717 | 0.5108 | 0.4041 | 0.5092 | 0.5156 | 0.6203 |

### 5.2 进行中：seed=2024 / 2025

- 已启动 6 个后台任务，分别跑 seed=2024、2025 的全部 34 配置（A-V-G+P 分 2 组 + A-V+P/G+P 1 组，各 seed 共 3 组）。
- 每完成一个 (track, subtrack, variant, seed) 即产出独立 JSON；完成后用第 4 节方法汇总为多 seed 文档。

### 5.3 已完成的工程优化（本轮）

- 特征预缓存 `_FEATURE_CACHE`：A-V+P 完整 5-fold 1888s→87s（22 倍），结果逐位一致。
- CUDA 不可用硬报错（train_cv.py / train.py / test.py 三处统一）。
- 各配置耗时实测：A-V+P ≈ 87s，A-V-G+P ≈ 437s，G+P ≈ 数十秒（缓存后）。

### 5.4 Git 提交历史（feature/champion-methods 分支，均已推送）

| 提交 | 内容 |
|---|---|
| `cf5024a` | perf: 特征预缓存 + CUDA 硬报错 |
| `ba442b2` / `18cafb0` | docs: K019 诊断 + 22 倍加速记录 |
| `6d99998` | fix: train.py/test.py CUDA 硬报错 |
| `a32f75a` / `7bc3db5` 等 | 6 篇论文对比实验与结论 |

---

## 6. 后续步骤

1. 等 seed=2024/2025 全部跑完。
2. 用聚合脚本（或手工）计算每个「方法×配置」的跨 seed 均值±std。
3. 写入 `docs/PAPER_COMPARISON_MULTISEED.md` 并提交推送。
4. 若某个方法在跨 seed 后仍稳定不敌 baseline，可正式下结论「该论文方法在本任务上无增益」。
