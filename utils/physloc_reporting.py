"""Tidy PhysLoc result rows, category summaries, and severity statistics."""
from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .physloc_metrics import SEVERITY_ORDER


TAXONOMY_FIELDS = ("family", "severity", "complexity", "condition", "difficulty")

# Fixed display order (not alphabetical) matching utils/physloc_metrics.py.
SPATIAL_REGIONS = ("violating_object", "active_violating_object", "active_violation",
                   "active_violation_visible", "expected_object", "causal_consequence")
TEMPORAL_WINDOWS = ("before", "active_visible", "active_hidden", "consequence")


def _pair_row(pair_uid: str, sample_uid: str, taxonomy: Dict[str, object],
              score: str, pair: Dict[str, object]) -> Dict[str, object]:
    """Convert a valid/invalid denoising-loss comparison to one row."""
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
    """Flatten nested PhysLoc results into analysis-friendly records."""
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


def _localization_row(pair_uid: str, sample_uid: str,
                      taxonomy: Dict[str, object], score: str,
                      metric: Dict[str, object], kind: str
                      ) -> Dict[str, object]:
    """Convert a nullable localization metric to one tidy result row."""
    return {
        "pair_uid": pair_uid, "sample_uid": sample_uid,
        **{key: taxonomy.get(key) for key in TAXONOMY_FIELDS},
        "kind": kind, "score": score, "value": metric.get("value"),
        "valid_ppe": None, "invalid_ppe": None, "ppe_gap": None,
        "detected": None, "available": bool(metric.get("available")),
        "reason": metric.get("reason"),
    }


def _summary(values: Sequence[float]) -> Dict[str, object]:
    """Return count, mean, sample deviation, and normal 95% CI half-width."""
    a = np.asarray(values, np.float64)
    if not len(a):
        return {"count": 0, "mean": None, "std": None, "ci95": None}
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"count": int(len(a)), "mean": float(a.mean()), "std": std,
            "ci95": float(1.96 * std / math.sqrt(len(a)))}


def category_summaries(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate every available score over each PhysLoc taxonomy axis."""
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


def overall_summaries(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate every available score across the whole run, ignoring taxonomy."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get("available") and row.get("value") is not None:
            grouped[(row["score"], row["kind"])].append(row)
    out = []
    for (score, kind), items in sorted(grouped.items(), key=lambda x: str(x[0])):
        values = [float(row["value"]) for row in items]
        gaps = [float(row["ppe_gap"]) for row in items
                if row.get("ppe_gap") is not None]
        detected = [bool(row["detected"]) for row in items
                    if row.get("detected") is not None]
        out.append({
            "score": score, "kind": kind,
            "value": _summary(values), "ppe_gap": _summary(gaps),
            "detection_rate": (float(np.mean(detected)) if detected else None),
        })
    return out


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Assign average one-based ranks, including tied values."""
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
    """Compute rank correlation for severity-ordered score values."""
    if len(values) < 2:
        return None
    x = np.arange(1, len(values) + 1, dtype=np.float64)
    y = _rankdata(values)
    if y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def severity_sensitivity(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Matched weak/medium/strong ordering based only on unweighted PPE gaps."""
    matched = defaultdict(lambda: defaultdict(list))
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
        matched[key][severity].append(float(row["ppe_gap"]))

    group_rows = []
    for (pair_uid, family, score), observations in sorted(matched.items()):
        # Multiple clips can legitimately share a matched key and severity.
        # Average replicates instead of silently retaining the last row.
        ladder = {name: float(np.mean(values))
                  for name, values in observations.items()}
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
            "replicate_counts": {name: len(values)
                                 for name, values in observations.items()},
            "comparisons": comparisons,
            "full_ladder": (ladder["weak"] < ladder["medium"] < ladder["strong"]
                            if len(ladder) == 3 else None),
            "spearman": _spearman([ladder[name] for name in ordered]),
        })

    def summarize(items: Sequence[Dict[str, object]], score: str,
                  family: Optional[str] = None) -> Dict[str, object]:
        """Summarize ordering accuracy for one score/family selection."""
        comparison = {}
        for key in ("medium>weak", "strong>medium", "strong>weak"):
            values = [row["comparisons"][key] for row in items
                      if row["comparisons"][key] is not None]
            comparison[key] = {"count": len(values),
                               "accuracy": float(np.mean(values)) if values else None}
        full = [row["full_ladder"] for row in items if row["full_ladder"] is not None]
        correlations = [row["spearman"] for row in items if row["spearman"] is not None]
        return {
            "score": score, "ordered_pair_accuracy": comparison,
            "family": family,
            "full_ladder": {"count": len(full),
                            "accuracy": float(np.mean(full)) if full else None},
            "spearman": _summary(correlations),
        }

    by_score = [summarize(
        [row for row in group_rows if row["score"] == score], score)
        for score in sorted({row["score"] for row in group_rows})]
    keys = sorted({(row["family"], row["score"]) for row in group_rows},
                  key=str)
    by_family_score = [summarize(
        [row for row in group_rows
         if row["family"] == family and row["score"] == score],
        score, family) for family, score in keys]
    return {"matched_groups": group_rows, "by_score": by_score,
            "by_family_score": by_family_score}


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write tidy per-sample metrics using a stable column order."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ("pair_uid", "sample_uid") + TAXONOMY_FIELDS + (
        "kind", "score", "value", "valid_ppe", "invalid_ppe", "ppe_gap",
        "detected", "available", "reason")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _safe(value: object) -> str:
    """Convert an arbitrary label to a filesystem-safe component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _write_category_csv(path: str,
                        summaries: Sequence[Dict[str, object]]) -> None:
    """Write flattened category summaries to CSV."""
    fields = ("dimension", "category", "score", "kind", "count", "mean",
              "std", "ci95", "gap_count", "gap_mean", "gap_std", "gap_ci95",
              "detection_rate", "misrank_rate")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            value, gap = item["value"], item["ppe_gap"]
            writer.writerow({
                "dimension": item["dimension"], "category": item["category"],
                "score": item["score"], "kind": item["kind"],
                "count": value["count"], "mean": value["mean"],
                "std": value["std"], "ci95": value["ci95"],
                "gap_count": gap["count"], "gap_mean": gap["mean"],
                "gap_std": gap["std"], "gap_ci95": gap["ci95"],
                "detection_rate": item["detection_rate"],
                "misrank_rate": item["misrank_rate"],
            })


def _write_severity_csv(path: str, severity: Dict[str, object]) -> None:
    """Write matched severity ladders to CSV."""
    fields = ("pair_uid", "family", "score", "weak_gap", "medium_gap",
              "strong_gap", "medium_gt_weak", "strong_gt_medium",
              "strong_gt_weak", "full_ladder", "spearman")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in severity["matched_groups"]:
            writer.writerow({
                "pair_uid": item["pair_uid"], "family": item["family"],
                "score": item["score"],
                "weak_gap": item["gaps"].get("weak"),
                "medium_gap": item["gaps"].get("medium"),
                "strong_gap": item["gaps"].get("strong"),
                "medium_gt_weak": item["comparisons"]["medium>weak"],
                "strong_gt_medium": item["comparisons"]["strong>medium"],
                "strong_gt_weak": item["comparisons"]["strong>weak"],
                "full_ladder": item["full_ladder"],
                "spearman": item["spearman"],
            })


def _pyplot():
    """Load a headless pyplot backend with writable cache locations."""
    os.environ.setdefault("MPLCONFIGDIR", "/data/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/data/tmp/cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_category_breakdown(summaries: Sequence[Dict[str, object]],
                             plot_dir: str) -> None:
    """Plot the base denoising-error gap by taxonomy category, one figure."""
    panels = []
    for dimension in ("severity", "complexity", "condition", "difficulty"):
        rows = [row for row in summaries
                if row["dimension"] == dimension and row["score"] == "base_ppe"
                and row["ppe_gap"]["mean"] is not None]
        if len(rows) >= 2:
            panels.append((dimension, rows))
    if not panels:
        return
    plt = _pyplot()
    ncols = min(2, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4 * nrows),
                             squeeze=False)
    for index, (dimension, rows) in enumerate(panels):
        ax = axes[index // ncols][index % ncols]
        labels = [str(row["category"]) for row in rows]
        means = [row["ppe_gap"]["mean"] for row in rows]
        errors = [row["ppe_gap"]["ci95"] or 0 for row in rows]
        ax.bar(labels, means, yerr=errors, color="#4c78a8", capsize=4)
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set(title="by %s" % dimension, ylabel="invalid - valid denoising error")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.25)
    for index in range(len(panels), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.suptitle("Denoising-error gap by category")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "category_breakdown.png"), dpi=170)
    plt.close(fig)


def _plot_spatial_localization(overall: Sequence[Dict[str, object]],
                               plot_dir: str) -> None:
    """Plot region ranking (AP) and denoising-error gap by spatial region."""
    ap_by_region = {row["score"][:-3]: row for row in overall
                    if row["kind"] == "spatiotemporal_localization"
                    and row["score"].endswith("_ap")
                    and row["value"]["mean"] is not None}
    gap_by_region = {row["score"].split(":", 1)[1]: row for row in overall
                     if row["kind"] == "pair_ppe"
                     and str(row["score"]).startswith("spatial:")
                     and row["ppe_gap"]["mean"] is not None}
    ap_regions = [name for name in SPATIAL_REGIONS if name in ap_by_region]
    gap_regions = [name for name in SPATIAL_REGIONS if name in gap_by_region]
    if not ap_regions and not gap_regions:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    if ap_regions:
        means = [ap_by_region[name]["value"]["mean"] for name in ap_regions]
        errors = [ap_by_region[name]["value"]["ci95"] or 0 for name in ap_regions]
        axes[0].bar(ap_regions, means, yerr=errors, color="#59a14f", capsize=4)
        axes[0].axhline(0.5, color="#333333", linewidth=.8, linestyle="--",
                        label="chance")
        axes[0].set(title="Spatial ranking (AP)", ylabel="average precision",
                   ylim=(0, 1))
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].grid(axis="y", alpha=.25)
        axes[0].legend()
    else:
        axes[0].axis("off")
    if gap_regions:
        means = [gap_by_region[name]["ppe_gap"]["mean"] for name in gap_regions]
        errors = [gap_by_region[name]["ppe_gap"]["ci95"] or 0 for name in gap_regions]
        axes[1].bar(gap_regions, means, yerr=errors, color="#4c78a8", capsize=4)
        axes[1].axhline(0, color="#333333", linewidth=.8)
        axes[1].set(title="Denoising-error gap by region",
                   ylabel="invalid - valid denoising error")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(axis="y", alpha=.25)
    else:
        axes[1].axis("off")
    fig.suptitle("Spatial localization")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "spatial_localization.png"), dpi=170)
    plt.close(fig)


def _plot_temporal_localization(overall: Sequence[Dict[str, object]],
                                plot_dir: str) -> None:
    """Plot denoising-error gap by temporal window and event/consequence AP."""
    gap_by_window = {row["score"].split(":", 1)[1]: row for row in overall
                     if row["kind"] == "pair_ppe"
                     and str(row["score"]).startswith("temporal:")
                     and row["ppe_gap"]["mean"] is not None}
    ap_names = ("temporal_event_ap", "temporal_consequence_ap")
    ap_by_name = {row["score"]: row for row in overall
                 if row["kind"] == "temporal_localization"
                 and row["score"] in ap_names
                 and row["value"]["mean"] is not None}
    windows = [name for name in TEMPORAL_WINDOWS if name in gap_by_window]
    if not windows and not ap_by_name:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    if windows:
        means = [gap_by_window[name]["ppe_gap"]["mean"] for name in windows]
        errors = [gap_by_window[name]["ppe_gap"]["ci95"] or 0 for name in windows]
        axes[0].bar(windows, means, yerr=errors, color="#4c78a8", capsize=4)
        axes[0].axhline(0, color="#333333", linewidth=.8)
        axes[0].set(title="Denoising-error gap by window",
                   ylabel="invalid - valid denoising error")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(axis="y", alpha=.25)
    else:
        axes[0].axis("off")
    if ap_by_name:
        labels = [name for name in ap_names if name in ap_by_name]
        means = [ap_by_name[name]["value"]["mean"] for name in labels]
        errors = [ap_by_name[name]["value"]["ci95"] or 0 for name in labels]
        ticks = [name.replace("temporal_", "").replace("_ap", "") for name in labels]
        axes[1].bar(ticks, means, yerr=errors, color="#59a14f", capsize=4)
        axes[1].axhline(0.5, color="#333333", linewidth=.8, linestyle="--",
                        label="chance")
        axes[1].set(title="Temporal ranking (AP)", ylabel="average precision",
                   ylim=(0, 1))
        axes[1].grid(axis="y", alpha=.25)
        axes[1].legend()
    else:
        axes[1].axis("off")
    fig.suptitle("Temporal localization")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "temporal_localization.png"), dpi=170)
    plt.close(fig)


def _plot_severity_trends(rows: Sequence[Dict[str, object]],
                          severity: Dict[str, object], plot_dir: str) -> None:
    """Plot weak/medium/strong denoising-error-gap trends for headline scores."""
    if len({row.get("severity") for row in rows
           if row.get("severity") in SEVERITY_ORDER}) < 2:
        return
    headline_scores = ("base_ppe", "spatial:violating_object",
                       "spatial:causal_consequence", "temporal:active_visible",
                       "temporal:consequence")
    pair_rows = [row for row in rows if row["kind"] == "pair_ppe"
                and row.get("severity") in SEVERITY_ORDER
                and row.get("ppe_gap") is not None]
    present = [score for score in headline_scores
              if any(row["score"] == score for row in pair_rows)]
    if not present:
        return
    plt = _pyplot()
    labels = ["weak", "medium", "strong"]
    ncols = min(3, len(present))
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4 * nrows),
                             squeeze=False)
    for index, score in enumerate(present):
        ax = axes[index // ncols][index % ncols]
        selected = [row for row in pair_rows if row["score"] == score]
        summaries = [_summary([float(row["ppe_gap"]) for row in selected
                               if row["severity"] == label]) for label in labels]
        groups = [group for group in severity["matched_groups"]
                 if group["score"] == score]
        for group in groups:
            xs, ys = [], []
            for i, label in enumerate(labels):
                if label in group["gaps"]:
                    xs.append(i)
                    ys.append(group["gaps"][label])
            ax.plot(xs, ys, color="#999999", alpha=.18, linewidth=.8)
        means = [item["mean"] if item["mean"] is not None else np.nan
                for item in summaries]
        ci = [item["ci95"] if item["ci95"] is not None else 0 for item in summaries]
        ax.errorbar(range(3), means, yerr=ci, color="#d84a4a", marker="o",
                   linewidth=2, capsize=4)
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set_xticks(range(3), labels)
        ax.set_title(score)
        ax.grid(axis="y", alpha=.25)
    for index in range(len(present), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.suptitle("Severity trends (invalid - valid denoising error)")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "severity_trends.png"), dpi=170)
    plt.close(fig)


def _plot_severity_ordering(severity: Dict[str, object], plot_dir: str) -> None:
    """Plot full weak<medium<strong ladder accuracy, overall and by family."""
    ordering = [row for row in severity["by_score"]
               if row["full_ladder"]["accuracy"] is not None]
    family_rows = [row for row in severity.get("by_family_score", [])
                  if row["full_ladder"]["accuracy"] is not None]
    if not ordering and not family_rows:
        return
    plt = _pyplot()
    ncols = 2 if (ordering and family_rows) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols,
                             max(3, .45 * max(len(ordering), 1) + 1.5)))
    axes = np.atleast_1d(axes)
    index = 0
    if ordering:
        ax = axes[index]
        index += 1
        labels = [row["score"] for row in ordering]
        values = [row["full_ladder"]["accuracy"] for row in ordering]
        ax.barh(range(len(labels)), values, color="#59a14f")
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlim=(0, 1), xlabel="fraction weak < medium < strong",
              title="Full ladder accuracy by score")
        ax.grid(axis="x", alpha=.25)
    if family_rows:
        ax = axes[index]
        families = sorted({row["family"] for row in family_rows})
        scores = sorted({row["score"] for row in family_rows})
        matrix = np.full((len(scores), len(families)), np.nan)
        for row in family_rows:
            matrix[scores.index(row["score"]), families.index(row["family"])] = (
                row["full_ladder"]["accuracy"])
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1,
                          aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(families)), families, rotation=40, ha="right")
        ax.set_yticks(range(len(scores)), scores)
        ax.set_title("Accuracy by violation family")
        fig.colorbar(image, ax=ax, label="fraction weak < medium < strong")
    fig.suptitle("Severity ordering accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "severity_ordering.png"), dpi=170)
    plt.close(fig)


def write_analysis_bundle(run_dir: str, model: str,
                          rows: Sequence[Dict[str, object]],
                          categories: Sequence[Dict[str, object]],
                          severity: Dict[str, object]) -> List[str]:
    """Write analysis CSVs and best-effort plots.

    CSV failures remain fatal because they indicate an invalid output path or
    data contract. Plot failures are returned as warnings so optional figures
    cannot discard an otherwise expensive evaluation run.
    """
    data_dir = os.path.join(run_dir, "analysis", "data")
    plot_dir = os.path.join(run_dir, "analysis", "plots", _safe(model))
    # Clear stale PNGs (e.g. from an earlier plot layout) so the directory
    # only ever reflects the current bundle, not a mix of old and new files.
    if os.path.isdir(plot_dir):
        for name in os.listdir(plot_dir):
            if name.endswith(".png"):
                os.remove(os.path.join(plot_dir, name))
    os.makedirs(plot_dir, exist_ok=True)
    write_csv(os.path.join(data_dir, "metrics_%s.csv" % model), rows)
    _write_category_csv(os.path.join(data_dir, "category_summary_%s.csv" % model), categories)
    _write_severity_csv(os.path.join(data_dir, "severity_ladders_%s.csv" % model), severity)
    overall = overall_summaries(rows)
    warnings = []
    plot_jobs = (
        ("category breakdown", _plot_category_breakdown, (categories, plot_dir)),
        ("spatial localization", _plot_spatial_localization, (overall, plot_dir)),
        ("temporal localization", _plot_temporal_localization, (overall, plot_dir)),
        ("severity trends", _plot_severity_trends, (rows, severity, plot_dir)),
        ("severity ordering", _plot_severity_ordering, (severity, plot_dir)),
    )
    for label, function, arguments in plot_jobs:
        try:
            function(*arguments)
        except Exception as exc:  # plotting is optional; metrics are not
            warning = "%s were not written: %s" % (label, exc)
            print(warning)
            warnings.append(warning)
    return warnings
