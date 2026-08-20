"""多 seed 结果聚合：把每个「方法×配置」跨 seed 的 F1/Acc/Kappa 汇总。

扫描 experiments/cv_logs 下的 cv_result_5fold_*.json，
按 (variant, track, subtrack) 分组，对 seed∈{42,2024,2025} 求均值±标准差，
输出 Markdown 汇总表 + 缺失检查。
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

CV_LOG_ROOT = Path("experiments") / "cv_logs"
SEEDS = {42, 2024, 2025}
METHOD_ORDER = ["cross_fusion", "cvae", "ptmfim", "hope", "reliability", "hypergraph"]
CONFIG_ORDER = [
    ("Track1", "A-V+P"), ("Track1", "A-V-G+P"), ("Track1", "G+P"),
    ("Track2", "A-V+P"), ("Track2", "A-V-G+P"), ("Track2", "G+P"),
]
VARIANT_MAP = {"cls_only": "cross_fusion"}  # seed=42 早期 baseline 旧命名


def norm_variant(v: str) -> str:
    return VARIANT_MAP.get(v, v)


def file_date(name: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else ""


def fmt_mean_std(xs: list[float]) -> str:
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.4f}±(1 seed)"
    return f"{statistics.mean(xs):.4f}±{statistics.stdev(xs):.4f}"


def main() -> None:
    # key=(method, track, subtrack) -> {seed: (f1, acc, kappa, mtime)}
    data: dict[tuple[str, str, str], dict[int, tuple[float, float, float, float]]] = defaultdict(dict)

    for jf in sorted(CV_LOG_ROOT.rglob("cv_result_5fold_*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        if d.get("task") != "binary":
            continue
        if file_date(jf.name) < "2026-08-18":  # 只统计本次对比实验（08-18 起）
            continue
        seed = d.get("seed")
        if seed not in SEEDS:
            continue
        method = norm_variant(d.get("variant", "cross_fusion"))
        if method not in METHOD_ORDER:
            continue
        key = (method, d.get("track"), d.get("subtrack"))
        mtime = jf.stat().st_mtime
        f1 = d.get("cv_f1_mean")
        acc = d.get("cv_acc_mean")
        kappa = d.get("cv_kappa_mean")
        # 同 (key, seed) 取较新文件
        if seed not in data[key] or mtime > data[key][seed][3]:
            data[key][seed] = (f1, acc, kappa, mtime)

    # 缺失检查
    missing = []
    for m in METHOD_ORDER:
        for t, s in CONFIG_ORDER:
            if m == "cvae" and s == "G+P":
                continue
            for seed in SEEDS:
                if seed not in data.get((m, t, s), {}):
                    missing.append((m, t, s, seed))

    print("## 缺失检查")
    if missing:
        for m, t, s, seed in missing:
            print(f"- MISSING: {m} {t} {s} seed={seed}")
    else:
        print("全部方法×配置×seed 完整，无缺失。")

    # F1 主表
    print("\n## 跨 seed Macro-F1 均值±标准差（seed 42/2024/2025）\n")
    header = "| 方法 | " + " | ".join(f"{t} {s}" for t, s in CONFIG_ORDER) + " |"
    sep = "|---|" + "---|" * len(CONFIG_ORDER)
    print(header)
    print(sep)
    for m in METHOD_ORDER:
        cells = []
        for t, s in CONFIG_ORDER:
            if m == "cvae" and s == "G+P":
                cells.append("N/A")
                continue
            f1s = [data[(m, t, s)][seed][0] for seed in SEEDS if seed in data.get((m, t, s), {})]
            cells.append(fmt_mean_std(f1s))
        print(f"| {m} | " + " | ".join(cells) + " |")

    # 详细
    print("\n## 详细（每格 F1 / Acc / Kappa 跨 seed 均值±std）\n")
    for m in METHOD_ORDER:
        print(f"### {m}")
        for t, s in CONFIG_ORDER:
            if m == "cvae" and s == "G+P":
                continue
            seeds = [seed for seed in SEEDS if seed in data.get((m, t, s), {})]
            if not seeds:
                print(f"- {t} {s}: 无数据")
                continue
            f1s = [data[(m, t, s)][seed][0] for seed in seeds]
            accs = [data[(m, t, s)][seed][1] for seed in seeds]
            kps = [data[(m, t, s)][seed][2] for seed in seeds]
            print(f"- {t} {s}: F1 {fmt_mean_std(f1s)} / Acc {fmt_mean_std(accs)} / Kappa {fmt_mean_std(kps)}")


if __name__ == "__main__":
    main()
