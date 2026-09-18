"""Tidy PhysLoc result rows, category summaries, and severity statistics."""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .physloc_metrics import SEVERITY_ORDER


TAXONOMY_FIELDS = ("family", "severity", "complexity", "condition", "difficulty")


def _pair_row(pair_uid, sample_uid, taxonomy, score, pair):
    return {
        "pair_uid": pair_uid, "sample_uid": sample_uid,
        **{key: taxonomy.get(key) for key in TAXONOMY_FIELDS},
        "kind": "pair_ppe", "score": score,
        "value": pair.get("invalid_ppe"),
        "valid_ppe": pair.get("valid_ppe"),
        "invalid_ppe": pair.get("invalid_ppe"),
        "ppe_gap": pair.get("ppe_gap"), "detected": pair.get("detected"),
        "available": pair.get("ppe_gap") is not None,
        "reason": pair.get("reason"),
    }


def tidy_rows(results: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pair_uid, subgroup in results.items():
        for variation, samples in subgroup.items():
            if variation == "valid":
                continue
            for sample_uid, info in samples.items():
                taxonomy = info.get("taxonomy", {})
                base = (info.get("pair_metrics") or {}).get("base_ppe")
                if base:
                    rows.append(_pair_row(pair_uid, sample_uid, taxonomy,
                                          "base_ppe", base))
                for name, metric in (info.get("spatial_ppe") or {}).items():
                    if metric.get("pair"):
                        rows.append(_pair_row(pair_uid, sample_uid, taxonomy,
                                              "spatial:" + name, metric["pair"]))
                    if metric.get("severity_weighted_pair"):
                        rows.append(_pair_row(
                            pair_uid, sample_uid, taxonomy,
                            "spatial_severity_weighted:" + name,
                            metric["severity_weighted_pair"]))
                temporal = info.get("temporal_ppe") or {}
                for name, metric in ((temporal.get("pair") or {}).get("windows") or {}).items():
                    rows.append(_pair_row(pair_uid, sample_uid, taxonomy,
                                          "temporal:" + name, metric))
                contrast = ((temporal.get("pair") or {}).get(
                    "pair_relative_event_contrast") or {})
                if contrast:
                    rows.append({
                        "pair_uid": pair_uid, "sample_uid": sample_uid,
                        **{key: taxonomy.get(key) for key in TAXONOMY_FIELDS},
                        "kind": "temporal_localization",
                        "score": "pair_relative_event_contrast",
                        "value": contrast.get("value"), "valid_ppe": None,
                        "invalid_ppe": None, "ppe_gap": None, "detected": None,
                        "available": bool(contrast.get("available")),
                        "reason": contrast.get("reason"),
                    })
                for name in ("temporal_event_ap", "temporal_consequence_ap"):
                    metric = temporal.get(name)
                    if metric:
                        rows.append(_localization_row(
                            pair_uid, sample_uid, taxonomy, name, metric,
                            "temporal_localization"))
                for name, metric in (info.get("spatiotemporal_ppe") or {}).items():
                    rows.append(_localization_row(
                        pair_uid, sample_uid, taxonomy, name, metric,
                        "spatiotemporal_localization"))
    return rows


def _localization_row(pair_uid, sample_uid, taxonomy, score, metric, kind):
    return {
        "pair_uid": pair_uid, "sample_uid": sample_uid,
        **{key: taxonomy.get(key) for key in TAXONOMY_FIELDS},
        "kind": kind, "score": score, "value": metric.get("value"),
        "valid_ppe": None, "invalid_ppe": None, "ppe_gap": None,
        "detected": None, "available": bool(metric.get("available")),
        "reason": metric.get("reason"),
    }


def _summary(values: Sequence[float]) -> Dict[str, object]:
    a = np.asarray(values, np.float64)
    if not len(a):
        return {"count": 0, "mean": None, "std": None, "ci95": None}
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"count": int(len(a)), "mean": float(a.mean()), "std": std,
            "ci95": float(1.96 * std / math.sqrt(len(a)))}


def category_summaries(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for dimension in ("severity", "complexity", "condition", "difficulty"):
        grouped = defaultdict(list)
        for row in rows:
            if row.get("available") and row.get("value") is not None:
                grouped[(row["score"], row["kind"], row.get(dimension))].append(row)
        for (score, kind, category), items in sorted(grouped.items(), key=lambda x: str(x[0])):
            values = [float(row["value"]) for row in items]
            gaps = [float(row["ppe_gap"]) for row in items
                    if row.get("ppe_gap") is not None]
            detected = [bool(row["detected"]) for row in items
                        if row.get("detected") is not None]
            out.append({
                "dimension": dimension, "category": category,
                "score": score, "kind": kind,
                "value": _summary(values), "ppe_gap": _summary(gaps),
                "detection_rate": (float(np.mean(detected)) if detected else None),
                "misrank_rate": (float(1.0 - np.mean(detected)) if detected else None),
            })
    return out


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    x = np.arange(1, len(values) + 1, dtype=np.float64)
    y = _rankdata(values)
    if y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def severity_sensitivity(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Matched weak/medium/strong ordering based only on unweighted PPE gaps."""
    matched = defaultdict(dict)
    excluded_weighted = "spatial_severity_weighted:"
    for row in rows:
        if row.get("kind") != "pair_ppe" or row.get("ppe_gap") is None:
            continue
        if str(row["score"]).startswith(excluded_weighted):
            continue
        severity = row.get("severity")
        if severity not in SEVERITY_ORDER:
            continue
        key = (row["pair_uid"], row.get("family"), row["score"])
        matched[key][severity] = float(row["ppe_gap"])

    group_rows = []
    for (pair_uid, family, score), ladder in sorted(matched.items()):
        ordered = [name for name in ("weak", "medium", "strong") if name in ladder]
        comparisons = {}
        for lower, upper in (("weak", "medium"), ("medium", "strong"),
                             ("weak", "strong")):
            comparisons[upper + ">" + lower] = (
                ladder[upper] > ladder[lower]
                if lower in ladder and upper in ladder else None)
        group_rows.append({
            "pair_uid": pair_uid, "family": family, "score": score,
            "gaps": dict(ladder), "available_severities": ordered,
            "comparisons": comparisons,
            "full_ladder": (ladder["weak"] < ladder["medium"] < ladder["strong"]
                            if len(ladder) == 3 else None),
            "spearman": _spearman([ladder[name] for name in ordered]),
        })

    by_score = []
    for score in sorted({row["score"] for row in group_rows}):
        items = [row for row in group_rows if row["score"] == score]
        comparison = {}
        for key in ("medium>weak", "strong>medium", "strong>weak"):
            values = [row["comparisons"][key] for row in items
                      if row["comparisons"][key] is not None]
            comparison[key] = {"count": len(values),
                               "accuracy": float(np.mean(values)) if values else None}
        full = [row["full_ladder"] for row in items if row["full_ladder"] is not None]
        correlations = [row["spearman"] for row in items if row["spearman"] is not None]
        by_score.append({
            "score": score, "ordered_pair_accuracy": comparison,
            "full_ladder": {"count": len(full),
                            "accuracy": float(np.mean(full)) if full else None},
            "spearman": _summary(correlations),
        })
    return {"matched_groups": group_rows, "by_score": by_score}


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ("pair_uid", "sample_uid") + TAXONOMY_FIELDS + (
        "kind", "score", "value", "valid_ppe", "invalid_ppe", "ppe_gap",
        "detected", "available", "reason")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
