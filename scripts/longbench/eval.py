#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# ---------- Dataset order consistent with original script ----------
DATASET_ORDER = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

# ---------- Arguments ----------
def parse_args(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--longbench_e", action="store_true")
    return ap.parse_args(args)

# ---------- Scoring ----------
def scorer_e(dataset, preds, gts, lengths, all_classes):
    buckets = {"0-4k": [], "4-8k": [], "8k+": []}
    for p, A, L in zip(preds, gts, lengths):
        best = max(dataset2metric[dataset](
            p.lstrip("\n").split("\n")[0] if dataset in {"trec", "triviaqa", "samsum", "lsht"} else p,
            gt,
            all_classes=all_classes) for gt in A)
        key = "0-4k" if L < 4000 else "4-8k" if L < 8000 else "8k+"
        buckets[key].append(best)
    return {k: round(100 * np.mean(v), 2) if v else -1 for k, v in buckets.items()}

def scorer(dataset, preds, gts, all_classes):
    total = 0.0
    for p, A in zip(preds, gts):
        best = max(dataset2metric[dataset](
            p.lstrip("\n").split("\n")[0] if dataset in {"trec", "triviaqa", "samsum", "lsht"} else p,
            gt,
            all_classes=all_classes) for gt in A)
        total += best
    return round(100 * total / len(preds), 2)

# ---------- Main ----------
def main():
    args = parse_args()
    rd = args.results_dir
    if not os.path.isdir(rd):
        raise FileNotFoundError(rd)

    # 1. Build dataset→[(method,path)...]
    dataset2files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    method_set = set()

    for ds in DATASET_ORDER:
        ds_dir = os.path.join(rd, ds)
        if not os.path.isdir(ds_dir):
            continue
        for fname in os.listdir(ds_dir):
            if fname.endswith(".json") and fname != "metrics.json":
                method = os.path.splitext(fname)[0]
                method_set.add(method)
                dataset2files[ds].append((method, os.path.join(ds_dir, fname)))

    # 2. Score
    results: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: -1))
    for ds, files in dataset2files.items():
        for method, path in files:
            try:
                preds, ans, lens, all_cls = [], [], [], []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        preds.append(obj["pred"])
                        ans.append(obj["answers"])
                        all_cls = obj.get("all_classes", [])
                        if "length" in obj:
                            lens.append(obj["length"])
                if not preds:
                    continue
                score = scorer_e(ds, preds, ans, lens, all_cls) if args.longbench_e \
                        else scorer(ds, preds, ans, all_cls)
                results[method][ds] = score
                print(f"[✓] {ds:<16} {method:<20} -> {score}")
            except Exception as e:
                print(f"[✗] {ds:<16} {method:<20} failed: {e}")

    # 3. Write CSV: fill missing entries with -1
    header = ["method"] + DATASET_ORDER
    rows = [header]
    for method in sorted(results):
        # Only include methods that have at least one valid score (not -1)
        if any(results[method][ds] != -1 for ds in DATASET_ORDER):
            rows.append([method] + [results[method][ds] for ds in DATASET_ORDER])

    out_csv = os.path.join(rd, "results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows(rows)

    print(f"\nCSV written to {out_csv}, {len(rows) - 1} method records total")

if __name__ == "__main__":
    main()
