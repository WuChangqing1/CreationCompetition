# MPDD Eight-Model Independent Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible independent-test evaluator for the eight approved models that ensembles the five CV folds, reports event- and subject-level metrics for MPDD 2025 and 2026, and writes a new result CSV without modifying any existing CV result.

**Architecture:** Extend the dataset-layout adapter with strict independent-test resolvers, place reusable run selection/aggregation/atomic-output logic in a small core module, isolate PyTorch and classical five-fold inference in a model module, and expose one orchestration-only CLI. The CLI will resolve the exact CV run from `raw_results.csv`, generate all probabilities before consulting test labels for metrics, then atomically upsert only the independent-test artifacts.

**Tech Stack:** Python 3, pathlib/csv/json, NumPy, pandas, scikit-learn, PyTorch, XGBoost, matplotlib, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-07-independent-test-evaluation-design.md`

## Global Constraints

- Work only on branch `GuoChuang`; preserve unrelated user changes.
- Run project commands only in the activated Conda environment `dachuangxiangmu`.
- User-facing commands must use plain `python` after `conda activate dachuangxiangmu`; do not put `conda run` in the documentation.
- The approved models are exactly `svm,xgboost,mlp,bilstm,lightweighttrans,lmf,mult,our`.
- Do not start the full 2025 or 2026 independent-test evaluation while implementing this plan.
- Never modify `experiments/results/raw_results.csv`, `experiments/results/summary.csv`, or the legacy `answer_Track1/submission.csv`.
- Use the test labels only after the five fold probabilities have been produced. Do not tune a threshold, choose a fold, or weight folds from test performance.
- SVM runs on CPU. XGBoost and all six PyTorch models require CUDA for the formal evaluation and must fail instead of silently falling back to CPU.
- Keep output replacement atomic and keyed by `(DatasetYear, Cohort, Model, EvaluationLevel)`.

---

## Task 1: Resolve 2025 and 2026 independent-test datasets strictly

**Files:**

- Modify: `experiments/dataset_layouts.py`
- Create: `tests/test_independent_test_dataset.py`

- [ ] **Step 1: Write failing 2025 alignment tests**

Create temporary directories representing:

```text
2025/MPDD-Test/MPDD-Elderly/
  labels/Testing_files.json
  individualEmbedding/descriptions_embeddings_with_ids.npy
  1s/Audio/mfccs/1178_A_1.npy
  1s/Visual/densenet/1178_V_1.npy
2025/MPDD-Test/MM2025_Track1_Elderly.json
```

The manifest fixture must omit the label and contain the relative feature paths. The ground-truth fixture must contain `{"test_id": "1178", "label_bin": 1}`. Assert that:

```python
paths = resolve_independent_test(root, "2025", "Elder", "1s", "mfccs", "densenet")
self.assertEqual(paths["entries"][0]["subject_id"], "1178")
self.assertEqual(paths["entries"][0]["bin_category"], 1)
self.assertEqual(paths["audio"], audio_root)
self.assertEqual(paths["video"], video_root)
```

Add negative tests for a missing ground-truth ID, a duplicate ground-truth ID, and a manifest row whose audio/video stems do not identify the same subject. Each must name the offending ID or path in the exception.

- [ ] **Step 2: Write failing 2026 alignment tests**

Create a temporary 2026 test tree with `split_labels_test.csv`, subject directories under `Audio/mfcc` and `Video/densenet`, and two paired events. Assert that `mfccs` resolves to `mfcc`, both events inherit the subject's `label2`, and missing/unpaired events fail rather than being skipped.

- [ ] **Step 3: Run the focused tests and verify RED**

```powershell
python -m unittest tests.test_independent_test_dataset -v
```

Expected: import failure for `resolve_independent_test` or assertion failures because independent-test layout support does not yet exist.

- [ ] **Step 4: Implement strict independent-test resolution**

Add these helpers to `experiments/dataset_layouts.py`:

```python
def _subject_from_feature_path(path):
    return Path(path).name.split("_", 1)[0]


def _load_unique_2025_test_labels(label_path):
    rows = json.loads(Path(label_path).read_text(encoding="utf-8"))
    labels = {}
    for row in rows:
        subject_id = str(row["test_id"])
        if subject_id in labels:
            raise ValueError(f"Duplicate 2025 test label for subject {subject_id}")
        labels[subject_id] = int(row["label_bin"])
    return labels


def build_2025_test_entries(manifest_path, label_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    labels = _load_unique_2025_test_labels(label_path)
    entries = []
    for row in manifest:
        audio_subject = _subject_from_feature_path(row["audio_feature_path"])
        video_subject = _subject_from_feature_path(row["video_feature_path"])
        if audio_subject != video_subject:
            raise ValueError(
                f"2025 test A/V subject mismatch: {row['audio_feature_path']} vs {row['video_feature_path']}"
            )
        if audio_subject not in labels:
            raise KeyError(f"Missing 2025 test label for subject {audio_subject}")
        entries.append({**row, "subject_id": audio_subject, "bin_category": labels[audio_subject]})
    return entries
```

Add `resolve_independent_test(data_root, dataset_year, cohort, split_window, audio_feature, video_feature)` returning the same five-key contract as `resolve_dataset`: `entries`, `personality`, `audio`, `video`, `track_root`. Resolve the actual 2025 and 2026 folder names case-insensitively with `_find_child`; validate that every referenced feature file exists and that the A/V entry counts and IDs match exactly.

For 2026, tighten `build_2026_entries` with an optional strict mode or a dedicated wrapper so `set(audio_files) != set(video_files)` reports missing event numbers instead of silently intersecting them. Preserve current training behavior unless the existing training tests establish that strict pairing is already safe.

- [ ] **Step 5: Run dataset tests and existing layout regression tests**

```powershell
python -m unittest tests.test_independent_test_dataset tests.test_dataset_layouts -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add experiments/dataset_layouts.py tests/test_independent_test_dataset.py
git commit -m "feat: resolve MPDD independent test datasets"
```

---

## Task 2: Select the exact CV run and validate all five folds

**Files:**

- Create: `experiments/independent_test.py`
- Create: `tests/test_independent_test_core.py`

- [ ] **Step 1: Write failing run-selection tests**

Use in-memory dictionaries with all `RAW_COLUMNS`. Test the complete condition set and ensure values read from CSV are normalized safely:

```python
condition = {
    "DatasetYear": "2025",
    "Cohort": "Elder",
    "Model": "mlp",
    "Track": "Track1",
    "Task": "binary",
    "AudioFeature": "mfccs",
    "VideoFeature": "densenet",
    "UsePersonality": True,
    "SplitWindow": "1s",
    "Seed": 3407,
}
run_id, fold_rows = select_cv_run(rows, condition, folds=5)
self.assertEqual(run_id, "mlp-example")
self.assertEqual([int(row["Fold"]) for row in fold_rows], [1, 2, 3, 4, 5])
```

Add tests that reject:

- two candidate Run_ID values;
- folds `{1,2,3,4}`;
- duplicate fold rows;
- a row with `Status != PASS`;
- boolean spellings other than a correctly normalized true/false value;
- a different feature, seed, task, cohort, or year.

- [ ] **Step 2: Write failing checkpoint-path and metadata tests**

Test `resolve_checkpoint_paths(results_dir, run_id, model, folds)` against a temporary tree and require exactly five files. Test `validate_checkpoint_metadata(payload, expected)` for model, fold, seed, year, cohort, track, task, audio/video feature, personality flag, split window, and device family.

- [ ] **Step 3: Run focused tests and verify RED**

```powershell
python -m unittest tests.test_independent_test_core -v
```

- [ ] **Step 4: Implement normalized exact matching**

In `experiments/independent_test.py`, define:

```python
RESULT_COLUMNS = [
    "DatasetYear", "Cohort", "Model", "EvaluationLevel", "Samples",
    "Accuracy", "Macro_F1", "Weighted_F1", "Precision", "Recall",
    "Positive_Recall", "Specificity", "ROC_AUC", "TN", "FP", "FN", "TP",
    "EnsembleFolds", "Run_ID", "Device", "Seed", "Status",
]


def select_cv_run(rows, condition, folds=5):
    matching = [row for row in rows if _row_matches(row, condition) and row.get("Status") == "PASS"]
    run_ids = sorted({row["Run_ID"] for row in matching})
    if len(run_ids) != 1:
        raise ValueError(f"Expected one CV Run_ID for {condition}; candidates={run_ids}")
    selected = [row for row in matching if row["Run_ID"] == run_ids[0]]
    actual_folds = [int(row["Fold"]) for row in selected]
    expected_folds = list(range(1, folds + 1))
    if sorted(actual_folds) != expected_folds or len(actual_folds) != len(set(actual_folds)):
        raise ValueError(f"Incomplete or duplicate folds for {run_ids[0]}: {actual_folds}")
    return run_ids[0], sorted(selected, key=lambda row: int(row["Fold"]))
```

Normalize only known scalar representations; never use truthiness such as `bool("False")`. Keep the full condition in ambiguity errors.

Implement checkpoint path construction as:

```python
results_dir / "raw" / run_id / model / f"fold_{fold}" / "checkpoint.pth"
```

`validate_checkpoint_metadata` must inspect top-level `model_name`, `fold`, `seed`, and the nested `feature_config`. It must accumulate all mismatches and raise one readable `ValueError` before any inference.

- [ ] **Step 5: Run focused tests**

```powershell
python -m unittest tests.test_independent_test_core -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add experiments/independent_test.py tests/test_independent_test_core.py
git commit -m "feat: validate independent test CV provenance"
```

---

## Task 3: Implement fold averaging, subject aggregation, and atomic result upsert

**Files:**

- Modify: `experiments/independent_test.py`
- Modify: `tests/test_independent_test_core.py`

- [ ] **Step 1: Write failing probability and subject aggregation tests**

Use two small fold arrays whose expected mean is obvious. Assert shape checking, finite values, and row sums near one:

```python
mean = mean_fold_probabilities([
    np.array([[0.8, 0.2], [0.4, 0.6]]),
    np.array([[0.6, 0.4], [0.2, 0.8]]),
])
np.testing.assert_allclose(mean, [[0.7, 0.3], [0.3, 0.7]])
```

For subject aggregation, supply subjects `A,A,B`, labels `1,1,0`, and three probability rows. Assert that A's probability is the arithmetic mean of its two events. Add a negative test for conflicting labels within subject A.

- [ ] **Step 2: Write failing result-row and atomic-upsert tests**

Build metrics through `evaluate_predictions` and assert `TN,FP,FN,TP` are extracted in that order. In a temporary directory:

1. Write one 2025 event row.
2. Upsert one 2026 event row and verify both remain.
3. Upsert the same 2025 key and verify only that row changes.
4. Point a sentinel `raw_results.csv` at known bytes and verify those bytes are unchanged.
5. Simulate a write failure before replacement and verify the original output CSV remains readable.

- [ ] **Step 3: Run focused tests and verify RED**

```powershell
python -m unittest tests.test_independent_test_core -v
```

- [ ] **Step 4: Implement pure aggregation helpers**

Add the pure helpers `mean_fold_probabilities(fold_probabilities)`, `aggregate_subject_probabilities(subject_ids, labels, probabilities)`, and `build_result_row(*, year, cohort, model, level, labels, probabilities, folds, run_id, device, seed, status="PASS")`.

`build_result_row` must call `evaluate_predictions`, compute `argmax(axis=1)`, extract the 2x2 confusion matrix, and return exactly `RESULT_COLUMNS`. Use `"N/A"` for ROC-AUC only when the evaluator cannot calculate it.

- [ ] **Step 5: Implement atomic artifacts**

Add `RESULT_KEY = ("DatasetYear", "Cohort", "Model", "EvaluationLevel")`, `upsert_result_rows(path, new_rows)`, and `write_prediction_csv(path, subject_ids, labels, probabilities, model, level)`.

`upsert_result_rows` must read with `utf-8-sig`, reject malformed existing headers, replace only identical keys, write all `RESULT_COLUMNS` to `<name>.csv.tmp`, and call `Path.replace` only after the writer closes successfully. Sort deterministically by year, cohort, model, and level.

Prediction files must contain at least:

```text
subject_id,true_label,pred_label,prob_0,prob_1,model,evaluation_level
```

- [ ] **Step 6: Run focused tests**

```powershell
python -m unittest tests.test_independent_test_core -v
```

Expected: all tests pass and no temp file remains after success.

- [ ] **Step 7: Commit Task 3**

```powershell
git add experiments/independent_test.py tests/test_independent_test_core.py
git commit -m "feat: aggregate and atomically store independent results"
```

---

## Task 4: Implement the six-model PyTorch five-checkpoint ensemble

**Files:**

- Create: `experiments/independent_test_models.py`
- Create: `tests/test_independent_test_models.py`

- [ ] **Step 1: Write failing PyTorch ensemble tests**

Use `unittest.mock` only at the expensive model boundary. Supply a two-event dataset and five temporary checkpoint payloads. Patch model creation so each fold returns a deterministic two-column probability matrix, then assert:

- all five checkpoints are loaded exactly once;
- metadata validation happens before prediction;
- the output equals the arithmetic mean of all five folds;
- an inconsistent checkpoint fold, model name, seed, or feature config raises before any result artifact is written;
- CUDA unavailable with requested device `cuda` raises and does not fall back.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
python -m unittest tests.test_independent_test_models.IndependentTestTorchTests -v
```

- [ ] **Step 3: Implement the PyTorch inference path**

Reuse `load_model_config`, `build_opt`, `infer_feature_dims`, `make_dataset`, and `evaluate_torch` behavior from `experiments/run_model_cv.py`; do not duplicate feature transformations. Implement:

```python
def infer_torch_fold_ensemble(
    *, model_name, test_entries, test_paths, checkpoint_paths,
    args, expected_metadata,
):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal independent-test evaluation")
    fold_probabilities = []
    for fold, checkpoint_path in enumerate(checkpoint_paths, 1):
        payload = torch.load(checkpoint_path, map_location="cpu")
        validate_checkpoint_metadata(payload, {**expected_metadata, "fold": fold})
        opt = SimpleNamespace(**payload["config"])
        opt.isTrain = False
        opt.gpu_ids = []
        opt.device = str(torch.device(args.device))
        model = create_experiment_model(model_name, opt=opt)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(torch.device(args.device))
        _, _, probabilities = evaluate_torch(model, loader, torch.device(args.device))
        fold_probabilities.append(probabilities)
    return mean_fold_probabilities(fold_probabilities)
```

Construct one deterministic, non-shuffled test `DataLoader` and reuse it for each fold. Preserve entry order. Use pinned memory for CUDA. Confirm that model output is `[N,2]` and labels match the dataset labels, but do not pass labels into model selection or checkpoint loading.

- [ ] **Step 4: Run PyTorch model tests**

```powershell
python -m unittest tests.test_independent_test_models.IndependentTestTorchTests -v
```

Expected: all tests pass without a real GPU or real checkpoint.

- [ ] **Step 5: Commit PyTorch inference**

```powershell
git add experiments/independent_test_models.py tests/test_independent_test_models.py
git commit -m "feat: ensemble PyTorch CV checkpoints on test data"
```

---

## Task 5: Refit and ensemble SVM/XGBoost from the original folds

**Files:**

- Modify: `experiments/independent_test_models.py`
- Modify: `tests/test_independent_test_models.py`

- [ ] **Step 1: Write failing classical-refit tests**

Build a small synthetic training manifest with enough subjects for mocked fold records and a separate test dataset. Patch `create_experiment_model` with a recording estimator that implements `fit` and `predict_proba`. Assert:

- five independent estimators are created and fitted;
- each estimator receives only its corresponding `train_ids`, never `val_ids`;
- the test feature matrix is transformed once with `pool_multimodal_features` semantics;
- personality is included only when `use_personality=True`;
- the five outputs are averaged equally;
- SVM reports `cpu:svm` while XGBoost reports `cuda:xgboost`;
- XGBoost config contains `device="cuda"`, `tree_method="hist"`, and the original seed.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
python -m unittest tests.test_independent_test_models.IndependentTestClassicalTests -v
```

- [ ] **Step 3: Implement the classical inference path**

Add:

```python
def infer_classical_fold_ensemble(
    *, model_name, training_entries, training_paths, test_entries,
    test_paths, fold_records, config, args,
):
    test_dataset = make_dataset(test_entries, args, test_paths)
    test_a, test_v, test_p, test_y = collect_classical_arrays(test_dataset)
    test_x = pool_multimodal_features(test_a, test_v, test_p, args.use_personality)
    fold_probabilities = []
    for fold_record in fold_records:
        train_entries = filter_entries_by_subjects(training_entries, fold_record["train_ids"])
        train_dataset = make_dataset(train_entries, args, training_paths)
        train_a, train_v, train_p, train_y = collect_classical_arrays(train_dataset)
        train_x = pool_multimodal_features(train_a, train_v, train_p, args.use_personality)
        fold_config = {**config, "seed": args.seed}
        if model_name == "xgboost":
            fold_config.update(device=args.device, tree_method="hist")
        estimator = create_experiment_model(model_name, config=fold_config)
        estimator.fit(train_x, train_y)
        fold_probabilities.append(estimator.predict_proba(test_x))
    return test_y, mean_fold_probabilities(fold_probabilities)
```

Load the existing split records from `experiments/splits/<year>/<cohort>/fold_<n>.json`. Recompute the subject-fold fingerprint from current training entries and require it to equal every saved split record. Validate fold number, fold count, seed, label key, train/val disjointness, and complete five-fold coverage before fitting.

Do not serialize these newly fitted estimators unless a future approved design explicitly adds that requirement.

- [ ] **Step 4: Run all model tests**

```powershell
python -m unittest tests.test_independent_test_models -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit classical inference**

```powershell
git add experiments/independent_test_models.py tests/test_independent_test_models.py
git commit -m "feat: refit classical models for test ensembles"
```

---

## Task 6: Add the safe orchestration CLI

**Files:**

- Create: `experiments/run_independent_test.py`
- Create: `tests/test_independent_test_cli.py`

- [ ] **Step 1: Write failing argument and no-import-side-effect tests**

Test that importing the module does no work. Test `parse_args` using the exact comma-separated model string and assert:

```python
self.assertEqual(args.models, [
    "svm", "xgboost", "mlp", "bilstm",
    "lightweighttrans", "lmf", "mult", "our",
])
self.assertEqual(args.output.name, "independent_test_results.csv")
```

Reject unknown models, duplicate models, folds other than five for this evaluator, `--device cpu` when any non-SVM model is requested, and an output path resolving to `raw_results.csv` or `summary.csv`.

- [ ] **Step 2: Write failing orchestration tests**

Use temporary result/data trees and patch only the inference calls. Assert that one successful model writes:

- one event row and one subject row to the new CSV;
- uniquely named event and subject prediction CSVs;
- uniquely named event and subject confusion artifacts;
- the selected `Run_ID`, device, seed, and `EnsembleFolds=5`.

Then force the first model to fail and the second to succeed. Assert that the batch continues, no fake metrics are produced for the failed model, and a FAILED row contains a concise error in `Status` without changing existing successful rows. Keep the schema at the approved fields by encoding the summary as `FAILED: <reason>` in `Status`.

- [ ] **Step 3: Run CLI tests and verify RED**

```powershell
python -m unittest tests.test_independent_test_cli -v
```

- [ ] **Step 4: Implement argument parsing and safety guards**

Provide these CLI parameters:

```text
--models --dataset-year --cohort --track --task --data-root
--cv-results --output --audio-feature --video-feature
--use-personality/--no-use-personality --split-window --folds
--seed --device --feature-max-len --batch-size --splits-dir --results-dir
```

Defaults:

```python
ROOT / "experiments" / "results" / "raw_results.csv"
ROOT / "experiments" / "results" / "independent_test_results.csv"
ROOT / "experiments" / "splits" / year / cohort
```

Resolve all paths before comparing protected files so alternate spellings cannot bypass the guard.

- [ ] **Step 5: Implement one-model orchestration**

The per-model flow in `evaluate_one_model(args, model_name, cv_rows, training_paths, test_paths, fold_records)` must be explicit: build the complete CV condition; select the unique Run_ID; resolve and validate five checkpoints for a PyTorch model or validate and refit the five classical splits; obtain one `[N,2]` averaged probability matrix; only then read the test labels; build the event row; aggregate subject probabilities; build the subject row; and return both rows plus both prediction payloads and both confusion matrices. Set the recorded device to `cuda` for PyTorch, `cpu:svm` for SVM, or `cuda:xgboost` for XGBoost.

Important ordering: the model inference functions receive entries containing labels because `AudioVisualDataset` requires them, but orchestration must not evaluate, inspect class-specific test scores, or branch on labels until the final probability matrix exists. Add a code comment documenting this boundary.

- [ ] **Step 6: Implement batch continuation and artifact writes**

For each model, write predictions and confusion matrices only after both event and subject calculations succeed. Then atomically upsert both result rows together. If a model fails, print `[FAIL] <model>: <reason>`, upsert a FAILED event and subject row with `Samples` and metric fields set to `N/A`, and continue. Return a non-zero process exit status after the loop if any model failed so automation can detect partial failure.

Print `[PASS] <model>` for successful models and finally print the absolute output CSV path.

- [ ] **Step 7: Run CLI tests and help smoke test**

```powershell
python -m unittest tests.test_independent_test_cli -v
python experiments\run_independent_test.py --help
```

Expected: tests pass; help lists every approved argument; no real data is loaded.

- [ ] **Step 8: Commit the CLI**

```powershell
git add experiments/run_independent_test.py tests/test_independent_test_cli.py
git commit -m "feat: add eight-model independent test CLI"
```

---

## Task 7: Add an end-to-end synthetic smoke test

**Files:**

- Create: `tests/test_independent_test_smoke.py`
- Modify if needed: `experiments/run_independent_test.py`

- [ ] **Step 1: Write a temporary-directory smoke test**

Create a complete synthetic train/test layout, a synthetic five-fold split cache, one matching five-row CV run, and lightweight mocked probabilities. Call `main(argv)` with an explicit argument list for one torch-style model and one classical-style model. Do not use the repository's real `experiments/results` directory.

Verify:

- four PASS rows exist: two models times two levels;
- prediction rows preserve test-entry order;
- subject aggregation reduces multiple events to one subject row;
- the confusion matrices contain the expected four cells;
- no `raw_results.csv`, `summary.csv`, or legacy submission bytes changed;
- rerunning the same command leaves four rows, not eight;
- changing year from 2025 to 2026 preserves both years.

- [ ] **Step 2: Run smoke and regression tests**

```powershell
python -m unittest tests.test_independent_test_smoke -v
python -m unittest tests.test_orchestration tests.test_cv_support tests.test_evaluator -v
```

Expected: all pass. This test must not touch real datasets, train real models, or require CUDA.

- [ ] **Step 3: Commit the smoke test**

```powershell
git add tests/test_independent_test_smoke.py experiments/run_independent_test.py
git commit -m "test: cover independent test workflow end to end"
```

---

## Task 8: Document exact 2025 and 2026 commands and output interpretation

**Files:**

- Modify: `docs/08_运行指南与实验结果.md`
- Modify: `docs/CURRENT_STATUS.md`
- Modify: `docs/CHANGELOG.md`
- Modify: `PROJECT_CONTEXT.md`

- [ ] **Step 1: Add the two formal commands**

Document the prerequisite once:

```powershell
conda activate dachuangxiangmu
cd D:\CodingData\Competition\GuoChuang\MPDD-main
```

Document 2025:

```powershell
python experiments\run_independent_test.py --models "svm,xgboost,mlp,bilstm,lightweighttrans,lmf,mult,our" --dataset-year 2025 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2025" --cv-results "experiments\results\raw_results.csv" --output "experiments\results\independent_test_results.csv" --audio-feature mfccs --video-feature densenet --use-personality --split-window 1s --folds 5 --seed 3407 --device cuda
```

Document 2026:

```powershell
python experiments\run_independent_test.py --models "svm,xgboost,mlp,bilstm,lightweighttrans,lmf,mult,our" --dataset-year 2026 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2026" --cv-results "experiments\results\raw_results.csv" --output "experiments\results\independent_test_results.csv" --audio-feature mfccs --video-feature densenet --use-personality --split-window 1s --folds 5 --seed 3407 --device cuda
```

State clearly that running 2026 after 2025 appends/replaces only matching independent-test keys; it does not erase 2025.

- [ ] **Step 2: Document all output locations and reading rules**

Explain:

- main metrics: `experiments/results/independent_test_results.csv`;
- event predictions: `experiments/results/independent_test_predictions/<year>_<cohort>_<model>_event.csv`;
- subject predictions: same directory with `_subject.csv`;
- confusion matrices: `experiments/results/independent_test_confusion_matrix/`;
- `EvaluationLevel=event` counts event clips;
- `EvaluationLevel=subject` counts people and is the primary person-level comparison;
- `TN,FP,FN,TP` correspond to `[[TN,FP],[FN,TP]]`;
- `raw_results.csv` remains the five-fold validation record and must not be confused with independent-test results.

- [ ] **Step 3: Update persistent context/status/changelog**

Record the new entry point, protected files, test-output paths, five-fold ensemble rule, formal device behavior, and the fact that no full independent-test run was launched during implementation.

- [ ] **Step 4: Check documentation commands against parser help**

```powershell
python experiments\run_independent_test.py --help
rg -n "run_independent_test|independent_test_results|raw_results" docs PROJECT_CONTEXT.md
```

- [ ] **Step 5: Commit documentation**

```powershell
git add docs/08_运行指南与实验结果.md docs/CURRENT_STATUS.md docs/CHANGELOG.md PROJECT_CONTEXT.md
git commit -m "docs: explain independent test evaluation"
```

---

## Task 9: Final verification without running the formal evaluation

**Files:**

- Verify only; modify files only to fix discovered defects.

- [ ] **Step 1: Run all independent-test tests**

```powershell
python -m unittest tests.test_independent_test_dataset tests.test_independent_test_core tests.test_independent_test_models tests.test_independent_test_cli tests.test_independent_test_smoke -v
```

- [ ] **Step 2: Run the entire repository test suite**

```powershell
python -m unittest discover -s tests -v
```

- [ ] **Step 3: Verify the formal GPU environment without training**

```powershell
python -c "import torch, xgboost; print('torch_cuda=', torch.cuda.is_available()); print('cuda_count=', torch.cuda.device_count()); print('xgboost=', xgboost.__version__)"
```

Expected for the user's formal run: `torch_cuda=True`, at least one CUDA device, and an importable XGBoost. This command does not start evaluation.

- [ ] **Step 4: Verify protected artifacts and output paths**

```powershell
git diff -- experiments/results/raw_results.csv experiments/results/summary.csv answer_Track1/submission.csv
git status --short
```

Expected: no diff for any protected result. Generated test artifacts must exist only in temporary test directories.

- [ ] **Step 5: Review against the approved design**

Check every section of `docs/superpowers/specs/2026-09-07-independent-test-evaluation-design.md` and explicitly verify:

- both dataset layouts;
- unique full-condition Run_ID selection;
- complete five-fold validation;
- six checkpoint ensembles plus two classical refits;
- event and subject metrics;
- atomic upsert semantics;
- failure status and batch continuation;
- CUDA enforcement;
- protected old outputs;
- no formal long-running evaluation.

- [ ] **Step 6: Commit any verification fixes, then push**

If verification required changes, stage the known implementation and test files that were corrected:

```powershell
git add experiments/dataset_layouts.py experiments/independent_test.py experiments/independent_test_models.py experiments/run_independent_test.py tests/test_independent_test_dataset.py tests/test_independent_test_core.py tests/test_independent_test_models.py tests/test_independent_test_cli.py tests/test_independent_test_smoke.py
git commit -m "fix: harden independent test evaluation"
```

Push using the user's proxy:

```powershell
git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push origin GuoChuang
```

- [ ] **Step 7: Final handoff**

Report:

- exact commit hash and branch;
- tests executed and pass counts;
- confirmation that no full independent-test run was started;
- the exact 2025 and 2026 commands from Task 8;
- every output location;
- how to distinguish CV validation results from independent-test event/subject results;
- any known runtime expectation or limitation, especially SVM CPU behavior and GPU data-loading CPU utilization.
