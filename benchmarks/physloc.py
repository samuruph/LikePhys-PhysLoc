"""PhysLoc release traversal and pair/localization metric attachment."""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from utils.physloc_metrics import (
    annotation_grids,
    localization_metrics,
    normalize_condition,
    normalize_difficulty,
    temporal_metrics,
)


NAME = "physloc"
FILTERS = ("family", "scenario", "level", "condition", "severity_bin")
ScoreVideo = Callable[[object, str, object], Tuple[object, Dict[str, object]]]


def resolve_root(args: object) -> str:
    """Resolve and validate a local PhysLoc v2/v3 release root."""
    if not args.physloc_root:
        if not args.physloc_hub_repo:
            raise ValueError(
                "--data physloc needs --physloc_root (a release on disk) or "
                "--physloc_hub_repo (one to download)")
        from huggingface_hub import snapshot_download

        local = os.path.join(
            args.physloc_cache, args.physloc_hub_repo.replace("/", "__"))
        args.physloc_root = snapshot_download(
            repo_id=args.physloc_hub_repo,
            repo_type="dataset",
            local_dir=local,
        )
        print("Downloaded %s to %s"
              % (args.physloc_hub_repo, args.physloc_root))
    args.physloc_prompt = None
    from utils.physloc_dataset import validate_release

    args.physloc_loader = validate_release(
        args.physloc_root, getattr(args, "physloc_loader", None))
    print("Validated PhysLoc schema v3 at %s using %s"
          % (args.physloc_root, args.physloc_loader))
    return str(args.physloc_root)


def _filter_choices(value: object) -> set:
    """Normalize one scalar or iterable filter value to comparable strings."""
    if isinstance(value, (str, int, float)) or value is None:
        return {None if value is None else str(value)}
    return {None if item is None else str(item) for item in value}


def _sample_matches(sample: object, filters: Dict[str, object]) -> bool:
    """Return whether a PhysLoc sample satisfies every requested filter."""
    return all(str(getattr(sample, name)) in _filter_choices(value)
               for name, value in filters.items())


def _sample_difficulty(sample: object) -> object:
    """Read difficulty from either current or older canonical loader samples.

    Newer loaders expose ``Sample.difficulty`` directly.  Older schema-v3
    loaders retain the same information in ``scene_info`` but omit the
    convenience property, so this fallback preserves compatibility without
    changing the dataset or its canonical loader.
    """
    direct = getattr(sample, "difficulty", None)
    if direct is not None:
        return direct
    scene_info = getattr(sample, "scene_info", {}) or {}
    if not isinstance(scene_info, dict):
        return None
    return (scene_info.get("difficulty_analysis")
            or scene_info.get("difficulty"))


def iter_groups(root: str, split: Optional[str] = None,
                loader_path: Optional[str] = None,
                **filters: object) -> Iterable[Tuple[object, list]]:
    """Yield valid/invalid evaluator groups from a schema-v3 release."""
    from utils.physloc_dataset import open_dataset

    requested = {key: value for key, value in filters.items()
                 if value is not None}
    unknown = sorted(set(requested) - set(FILTERS))
    if unknown:
        raise TypeError("unknown PhysLoc filter(s) %s; known: %s"
                        % (unknown, list(FILTERS)))
    dataset = open_dataset(
        root, loader_path=loader_path, unit="pair", split=split,
        fields=("observations.rgb_path",))
    for pair in dataset.pairs():
        invalids = [sample for sample in pair.invalids
                    if _sample_matches(sample, requested)]
        if not invalids:
            continue
        clips = [("valid", pair.valid, pair.valid.video_path)]
        clips.extend(
            ("%s_%s" % (sample.family, sample.severity_bin),
             sample, sample.video_path)
            for sample in invalids)
        try:
            yield pair, clips
        finally:
            for _, sample, _ in clips:
                sample.release()


def _value(metric: object) -> Optional[float]:
    """Extract a nullable value from a packaged metric."""
    return metric.get("value") if isinstance(metric, dict) else None


def _pair_comparison(valid_value: Optional[float],
                     invalid_value: Optional[float]) -> Dict[str, object]:
    """Compare invalid PPE with valid-twin PPE using a shared region."""
    if valid_value is None or invalid_value is None:
        return {
            "valid_ppe": valid_value,
            "invalid_ppe": invalid_value,
            "ppe_gap": None,
            "detected": None,
            "reason": "pair_metric_unavailable",
        }
    gap = float(invalid_value - valid_value)
    return {
        "valid_ppe": float(valid_value),
        "invalid_ppe": float(invalid_value),
        "ppe_gap": gap,
        "detected": bool(gap > 0),
        "reason": None,
    }


def _temporal_pair_comparison(valid_metrics: Dict[str, object],
                              invalid_metrics: Dict[str, object]
                              ) -> Dict[str, object]:
    """Compare temporal windows and event elevation across a valid twin."""
    output: Dict[str, object] = {"windows": {}}
    for name in invalid_metrics["windows"]:
        valid_value = _value(valid_metrics["windows"].get(name, {}))
        invalid_value = _value(invalid_metrics["windows"].get(name, {}))
        output["windows"][name] = _pair_comparison(valid_value, invalid_value)

    valid_trace = np.asarray(valid_metrics["latent_frame_ppe"], np.float64)
    invalid_trace = np.asarray(invalid_metrics["latent_frame_ppe"], np.float64)
    active = np.asarray(invalid_metrics["latent_clocks"]["active"], bool)
    consequence = np.asarray(
        invalid_metrics["latent_clocks"]["consequence"], bool)
    positive = np.flatnonzero(active | consequence)
    before = np.arange(len(active)) < (int(positive[0]) if positive.size else 0)
    if active.any() and before.any():
        valid_elevation = float(
            valid_trace[active].mean() - valid_trace[before].mean())
        invalid_elevation = float(
            invalid_trace[active].mean() - invalid_trace[before].mean())
        output["pair_relative_event_contrast"] = {
            "value": invalid_elevation - valid_elevation,
            "invalid_elevation": invalid_elevation,
            "valid_elevation": valid_elevation,
            "available": True,
            "reason": None,
        }
    else:
        output["pair_relative_event_contrast"] = {
            "value": None,
            "available": False,
            "reason": "event_or_before_window_empty",
        }
    return output


def _attach_pair_metrics(args: object, valid_runtime: Dict[str, object],
                         invalid_runtime: Dict[str, object]) -> None:
    """Attach all requested pair and localization metrics to an invalid clip."""
    info = invalid_runtime["info"]
    info["pair_metrics"] = {
        "base_ppe": _pair_comparison(
            valid_runtime["info"]["loss"], info["loss"])
    }
    valid_error = valid_runtime["grid"]
    invalid_error = invalid_runtime["grid"]
    if valid_error is None or invalid_error is None:
        return
    if valid_error.shape != invalid_error.shape:
        raise ValueError("valid/invalid latent grids differ: %s vs %s"
                         % (valid_error.shape, invalid_error.shape))
    sample = invalid_runtime["sample"]
    indices = invalid_runtime["indices"]
    grids = annotation_grids(sample, indices, invalid_error.shape)
    info["alignment"] = {
        "latent_shape": list(invalid_error.shape),
        "sampled_frame_indices": list(indices),
        "annotation_frames": int(sample.num_frames),
        "annotation_loader": getattr(args, "physloc_loader", None),
    }

    if "temporal_ppe" in args.score_groups:
        invalid_temporal = temporal_metrics(invalid_error, sample, indices)
        valid_temporal = temporal_metrics(valid_error, sample, indices)
        invalid_temporal["valid_latent_frame_ppe"] = (
            valid_temporal["latent_frame_ppe"])
        invalid_temporal["valid_rgb_frame_ppe"] = (
            valid_temporal["rgb_frame_ppe"])
        invalid_temporal["pair"] = _temporal_pair_comparison(
            valid_temporal, invalid_temporal)
        info["temporal_ppe"] = invalid_temporal

    if {"spatial_ppe", "spatiotemporal_ppe"} & set(args.score_groups):
        invalid_local = localization_metrics(invalid_error, grids)
        valid_local = localization_metrics(valid_error, grids)
        if "spatial_ppe" in args.score_groups:
            spatial = invalid_local["spatial"]
            for name, metrics in spatial.items():
                valid_metrics = valid_local["spatial"][name]
                metrics["pair"] = _pair_comparison(
                    _value(valid_metrics["ppe"]), _value(metrics["ppe"]))
                if "severity_weighted_ppe" in metrics:
                    metrics["severity_weighted_pair"] = _pair_comparison(
                        _value(valid_metrics["severity_weighted_ppe"]),
                        _value(metrics["severity_weighted_ppe"]))
            info["spatial_ppe"] = spatial
        if "spatiotemporal_ppe" in args.score_groups:
            info["spatiotemporal_ppe"] = invalid_local["spatiotemporal"]


def evaluate(args: object, pipe: object, score_video: ScoreVideo
             ) -> Dict[str, object]:
    """Evaluate PhysLoc pairs with shared masks for valid/invalid scoring."""
    if not hasattr(args, "artifact_warnings"):
        args.artifact_warnings = []
    filters = {name: getattr(args, "physloc_" + name) for name in FILTERS}
    groups = iter_groups(
        args.physloc_root, split=args.physloc_split,
        loader_path=args.physloc_loader, **filters)
    results: Dict[str, object] = {}
    for index, (pair, clips) in tqdm(
            enumerate(groups), desc="Evaluating PhysLoc pairs"):
        args.subgroup_seed = args.seed + index
        args.physloc_prompt = pair.prompt
        subgroup_results: Dict[str, object] = {}
        runtime: Dict[str, Dict[str, object]] = {}
        for variation, sample, video_path in tqdm(
                clips, desc="Evaluating %s" % pair.pair_uid):
            loss, log_info = score_video(args, video_path, pipe)
            if loss is None:
                continue
            info = {
                "loss": loss,
                "noise_pred_mean": log_info["noise_pred_mean"],
                "true_noise_mean": log_info["true_noise_mean"],
                "loss_array": log_info["loss_array"],
                "sampled_frame_indices": log_info["sampled_frame_indices"],
                "source_num_frames": log_info["source_num_frames"],
                "taxonomy": {
                    "family": sample.family,
                    "scenario": sample.scenario,
                    "severity": sample.severity_bin,
                    "complexity": sample.level,
                    "condition": normalize_condition(sample.condition),
                    "difficulty": normalize_difficulty(_sample_difficulty(sample)),
                },
            }
            if "latent_shape" in log_info:
                info["latent_shape"] = log_info["latent_shape"]
            subgroup_results.setdefault(variation, {})[sample.uid] = info
            runtime[sample.uid] = {
                "sample": sample,
                "grid": log_info.get("_error_grid"),
                "indices": log_info["sampled_frame_indices"],
                "info": info,
            }

        valid_runtime = runtime.get(pair.valid.uid)
        if valid_runtime is not None:
            for _, invalid_sample, _ in clips[1:]:
                invalid_runtime = runtime.get(invalid_sample.uid)
                if invalid_runtime is not None:
                    _attach_pair_metrics(args, valid_runtime, invalid_runtime)
        if args.visualize:
            from utils.physloc_visualization import render_pair_artifacts

            try:
                render_pair_artifacts(args.run_dir, pair, runtime, args.model)
            except Exception as exc:  # visualizations must not discard metrics
                warning = "visualization failed for %s: %s" % (
                    pair.pair_uid, exc)
                print(warning)
                args.artifact_warnings.append(warning)
        if subgroup_results:
            results[pair.pair_uid] = subgroup_results
    return results
