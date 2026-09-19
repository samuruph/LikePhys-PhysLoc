"""Dataset-neutral aggregation for PPE evaluation result JSON files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from benchmarks.evaluators.common import compute_misrank_normalized


MODEL_NAMES = {
    "animatediff": "AnimateDiff",
    "animatediff_sdxl": "AnimateDiff SDXL",
    "modelscope": "ModelScope",
    "zeroscope": "ZeroScope",
    "cogvideox": "CogVideoX",
    "cogvideox-5b": "CogVideoX-5B",
    "cogvideox1.5-5b": "CogVideoX1.5-5B",
    "hunyuan_t2v": "Hunyuan T2V",
    "wan2.1-T2V-1.3b": "Wan2.1-T2V-1.3B",
    "wan2.1-T2V-14b": "Wan2.1-T2V-14B",
    "ltx": "LTX",
    "ltx-0.9.1": "LTX v0.9.1",
    "ltx-0.9.5": "LTX v0.9.5",
    "mochi": "Mochi",
}


def result_files(root: Path) -> Iterable[Path]:
    """Yield result JSON files beneath an experiment or dataset directory."""
    yield from sorted(root.rglob("results_*.json"))


def _model_name(path: Path) -> str:
    """Extract the model identifier from a standard evaluator filename."""
    return path.stem.removeprefix("results_")


def metric_values(document: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    """Read persisted mis-rank metrics or reconstruct legacy result files."""
    persisted = document.get("misrank_metrics")
    if isinstance(persisted, dict):
        return persisted
    scenes = document.get("scene_evaluations", {})
    return compute_misrank_normalized(scenes)


def collect_records(root: Path) -> List[Dict[str, object]]:
    """Collect one tidy record per model/dataset/variation result metric."""
    records: List[Dict[str, object]] = []
    for path in result_files(root):
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        for variation, metric in metric_values(document).items():
            value = metric.get("misrank_ratio")
            if value is None:
                continue
            records.append({
                "dataset": path.parent.name,
                "model": _model_name(path),
                "variation": variation,
                "misrank_ratio": float(value),
                "source": str(path),
            })
    return records


def summarize(records: Sequence[Mapping[str, object]],
              weighting: str) -> List[Dict[str, object]]:
    """Summarize variation-level mis-ranks by model and dataset.

    ``variation`` weights every invalid variation equally; ``dataset`` first
    averages each dataset and then weights datasets equally.
    """
    if weighting not in {"variation", "dataset"}:
        raise ValueError("weighting must be 'variation' or 'dataset'")
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["dataset"])].append(
            float(row["misrank_ratio"]))
    output = []
    for (model, dataset), values in sorted(grouped.items()):
        output.append({"model": model, "dataset": dataset,
                       "count": len(values), "misrank_ratio": float(np.mean(values))})
    overall: Dict[str, List[float]] = defaultdict(list)
    if weighting == "variation":
        for row in records:
            overall[str(row["model"])].append(float(row["misrank_ratio"]))
    else:
        for row in output:
            overall[str(row["model"])].append(float(row["misrank_ratio"]))
    for model, values in overall.items():
        output.append({"model": model, "dataset": "overall", "count": len(values),
                       "misrank_ratio": float(np.mean(values))})
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write rows using a stable schema appropriate to their record type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys()) if rows else ("model", "dataset", "count", "misrank_ratio")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> bool:
    """Write a compact cross-dataset model comparison plot when available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    overall = sorted((row for row in rows if row["dataset"] == "overall"),
                     key=lambda row: row["misrank_ratio"])
    if not overall:
        return False
    labels = [MODEL_NAMES.get(row["model"], row["model"]) for row in overall]
    values = [100.0 * row["misrank_ratio"] for row in overall]
    figure, axis = plt.subplots(figsize=(9, max(3, 0.45 * len(labels) + 1.5)))
    axis.barh(range(len(labels)), values, color="#4c78a8")
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set(xlabel="Mean mis-rank (%)", title="PPE evaluation summary")
    axis.grid(axis="x", alpha=.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def plot_category_summary(path: Path,
                          rows: Sequence[Mapping[str, object]]) -> bool:
    """Write one grouped histogram with overall and all categories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    category_rows = [row for row in rows if row["dataset"] != "overall"]
    if not category_rows:
        return False
    models = sorted({str(row["model"]) for row in category_rows})
    datasets = sorted({str(row["dataset"]) for row in category_rows})
    categories = ["overall"] + datasets
    figure, axis = plt.subplots(
        figsize=(max(11, 0.85 * len(categories) + 4), 6.5))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(models))))
    positions = np.arange(len(categories), dtype=np.float64)
    bar_width = min(.8 / max(1, len(models)), .22)
    for model_index, model in enumerate(models):
        values = []
        for category in categories:
            selected = [row for row in rows
                        if row["dataset"] == category and
                        str(row["model"]) == model]
            values.append(100.0 * float(selected[0]["misrank_ratio"])
                          if selected else np.nan)
        offset = (model_index - (len(models) - 1) / 2.0) * bar_width
        bars = axis.bar(positions + offset, values, width=bar_width,
                        label=MODEL_NAMES.get(model, model),
                        color=colors[model_index])
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(bar.get_x() + bar.get_width() / 2,
                          min(98, value + 1.5), "%.1f" % value,
                          ha="center", va="bottom", fontsize=8,
                          rotation=90 if len(categories) > 8 else 0)
    axis.set_ylim(0, 100)
    axis.set_xticks(positions, categories, rotation=40, ha="right")
    axis.set_ylabel("Mis-rank (%)")
    axis.set_xlabel("Physics category / dataset")
    axis.set_title("PPE mis-rank: overall and per category")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def analyze_results(results_dir: str | Path,
                    output_dir: str | Path | None = None,
                    weighting: str = "variation") -> List[Dict[str, object]]:
    """Aggregate result JSON files and write dataset-neutral summary artifacts.

    Args:
        results_dir: An experiment directory, dataset directory, or parent
            directory containing evaluator ``results_*.json`` files.
        output_dir: Destination for CSV and PNG artifacts. Defaults to
            ``results_dir``.
        weighting: ``"variation"`` weights every invalid variation equally;
            ``"dataset"`` weights each dataset equally.

    Returns:
        The model/dataset and overall summary rows written to disk.

    Raises:
        FileNotFoundError: If no evaluator result files are found.
        ValueError: If ``weighting`` is unsupported.
    """
    root = Path(results_dir).resolve()
    records = collect_records(root)
    if not records:
        raise FileNotFoundError("no results_*.json files beneath %s" % root)
    destination = Path(output_dir).resolve() if output_dir else root
    summary = summarize(records, weighting)
    write_csv(destination / "misrank_by_variation.csv", records)
    write_csv(destination / "misrank_summary.csv", summary)
    plotted = plot_summary(destination / "misrank_summary.png", summary)
    category_plotted = plot_category_summary(
        destination / "misrank_by_category.png", summary)
    print("Read %d variation records from %d dataset(s)." %
          (len(records), len({row["dataset"] for row in records})))
    print("Wrote %s" % (destination / "misrank_summary.csv"))
    if plotted:
        print("Wrote %s" % (destination / "misrank_summary.png"))
    if category_plotted:
        print("Wrote %s" % (destination / "misrank_by_category.png"))
    for row in sorted((item for item in summary if item["dataset"] == "overall"),
                      key=lambda item: item["misrank_ratio"]):
        print("%-24s %.2f%%" %
              (MODEL_NAMES.get(row["model"], row["model"]),
               100.0 * row["misrank_ratio"]))
    return summary


def main(default_results_dir: Optional[str] = None) -> None:
    """Run command-line result aggregation for either benchmark or both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=default_results_dir or "results",
                        help="experiment directory or any parent containing results_*.json")
    parser.add_argument("--output-dir", default=None,
                        help="directory for CSV/PNG output (default: results directory)")
    parser.add_argument("--weighting", choices=("variation", "dataset"),
                        default="variation")
    args = parser.parse_args()
    analyze_results(args.results_dir, args.output_dir, args.weighting)


if __name__ == "__main__":
    main()
