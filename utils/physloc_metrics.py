"""Pure NumPy helpers for PhysLoc PPE localization and aggregation."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCORE_GROUPS = {"base_ppe", "temporal_ppe", "spatial_ppe",
                "spatiotemporal_ppe"}
SEVERITY_ORDER = {"weak": 1, "medium": 2, "strong": 3}


def normalize_condition(value: object) -> str:
    raw = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "standard": "standard", "default": "standard",
        "camera": "camera_motion", "camera_motion": "camera_motion", "motion": "camera_motion",
        "distractor": "distractors", "distractors": "distractors",
        "multi": "multi", "multiple": "multi",
        "multi_motion": "multi_motion", "multi_camera_motion": "multi_motion",
        "multi+motion": "multi_motion",
    }
    return aliases.get(raw, "other:" + raw)


def normalize_difficulty(value: object) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("level") or value.get("label")
    raw = str(value).strip().lower() if value is not None else ""
    aliases = {"medium": "moderate", "moderate": "moderate",
               "easy": "easy", "hard": "hard"}
    return aliases.get(raw)


def parse_score_groups(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Normalize repeatable and comma-separated ``--scores`` values."""
    raw: List[str] = []
    for value in values or ("base_ppe",):
        raw.extend(part.strip() for part in str(value).split(",") if part.strip())
    if "all" in raw:
        raw = sorted(SCORE_GROUPS)
    unknown = sorted(set(raw) - SCORE_GROUPS)
    if unknown:
        raise ValueError("unknown score group(s) %s; choose from %s or all"
                         % (unknown, sorted(SCORE_GROUPS)))
    return tuple(dict.fromkeys(raw or ("base_ppe",)))


def sampled_frame_indices(total_frames: int, requested_frames: int) -> np.ndarray:
    if total_frames <= 0 or requested_frames <= 0:
        raise ValueError("frame counts must be positive")
    return np.rint(np.linspace(0, total_frames - 1, requested_frames)).astype(int)


def temporal_bins(sampled_frames: int, latent_frames: int) -> List[np.ndarray]:
    """Map sampled RGB frames to latent frames, preserving 4k+1-style bins."""
    if sampled_frames <= 0 or latent_frames <= 0:
        raise ValueError("frame counts must be positive")
    if latent_frames == sampled_frames:
        return [np.asarray([i], int) for i in range(sampled_frames)]
    if latent_frames == 1:
        return [np.arange(sampled_frames)]
    if (sampled_frames - 1) % (latent_frames - 1) == 0:
        ratio = (sampled_frames - 1) // (latent_frames - 1)
        if ratio > 1:
            return [np.asarray([0], int)] + [
                np.arange(ratio * i - ratio + 1, ratio * i + 1)
                for i in range(1, latent_frames)
            ]
    return [np.asarray(part, int) for part in
            np.array_split(np.arange(sampled_frames), latent_frames)]


def _spatial_reduce(frame: np.ndarray, height: int, width: int,
                    reduction: str) -> np.ndarray:
    """Resize by reducing every source cell overlapping a target cell."""
    src_h, src_w = frame.shape
    out = np.zeros((height, width), np.float32)
    for y in range(height):
        y0 = int(np.floor(y * src_h / height))
        y1 = max(y0 + 1, int(np.ceil((y + 1) * src_h / height)))
        for x in range(width):
            x0 = int(np.floor(x * src_w / width))
            x1 = max(x0 + 1, int(np.ceil((x + 1) * src_w / width)))
            cell = frame[y0:min(y1, src_h), x0:min(x1, src_w)]
            out[y, x] = float(cell.max() if reduction == "max" else cell.mean())
    return out


def project_volume(value: np.ndarray, source_indices: Sequence[int],
                   target_shape: Tuple[int, int, int],
                   reduction: str = "max") -> np.ndarray:
    """Project a source ``[T,H,W]`` annotation onto an arbitrary latent grid."""
    source = np.asarray(value)
    if source.ndim != 3:
        raise ValueError("annotation must be [T,H,W], got %s" % (source.shape,))
    indices = np.clip(np.asarray(source_indices, int), 0, len(source) - 1)
    sampled = source[indices]
    target_t, target_h, target_w = map(int, target_shape)
    out = np.zeros(target_shape, np.float32)
    for t, frames in enumerate(temporal_bins(len(sampled), target_t)):
        chunk = sampled[frames]
        frame = chunk.max(axis=0) if reduction == "max" else chunk.mean(axis=0)
        out[t] = _spatial_reduce(np.asarray(frame, np.float32), target_h,
                                 target_w, reduction)
    return out


def project_mask(value: np.ndarray, source_indices: Sequence[int],
                 target_shape: Tuple[int, int, int]) -> np.ndarray:
    return project_volume(value, source_indices, target_shape, "max") > 0


def project_trace(value: np.ndarray, source_indices: Sequence[int],
                  latent_frames: int, reduction: str = "max") -> np.ndarray:
    source = np.asarray(value)
    indices = np.clip(np.asarray(source_indices, int), 0, len(source) - 1)
    sampled = source[indices]
    out = []
    for frames in temporal_bins(len(sampled), int(latent_frames)):
        chunk = sampled[frames]
        out.append(float(chunk.max() if reduction == "max" else chunk.mean()))
    return np.asarray(out)


def expand_latent_trace(trace: Sequence[float], sampled_frames: int) -> np.ndarray:
    trace = np.asarray(trace, np.float64)
    out = np.zeros(sampled_frames, np.float64)
    for index, frames in enumerate(temporal_bins(sampled_frames, len(trace))):
        out[frames] = trace[index]
    return out


def masked_mean(error: np.ndarray, mask: np.ndarray,
                weights: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[str]]:
    error = np.asarray(error, np.float64)
    selected = np.asarray(mask, bool)
    if error.shape != selected.shape:
        raise ValueError("error and mask shapes differ: %s vs %s"
                         % (error.shape, selected.shape))
    if not selected.any():
        return None, "empty_region"
    if weights is None:
        return float(error[selected].mean()), None
    weight = np.asarray(weights, np.float64) * selected
    total = float(weight.sum())
    if total <= 0:
        return None, "zero_severity_weight"
    return float((error * weight).sum() / total), None


def average_precision(scores: np.ndarray, positives: np.ndarray,
                      candidates: Optional[np.ndarray] = None
                      ) -> Tuple[Optional[float], Optional[str]]:
    """Average precision with stable tie handling and no sklearn dependency."""
    score = np.asarray(scores, np.float64)
    truth = np.asarray(positives, bool)
    domain = np.ones(truth.shape, bool) if candidates is None else np.asarray(candidates, bool)
    truth = truth & domain
    if not truth.any():
        return None, "no_positive_tokens"
    if not (domain & ~truth).any():
        return None, "no_negative_tokens"
    y = truth[domain]
    s = score[domain]
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(precision[y].mean()), None


def error_ratio(error: np.ndarray, target: np.ndarray,
                outside: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    inside, reason = masked_mean(error, target)
    if reason:
        return None, reason
    baseline, reason = masked_mean(error, outside)
    if reason:
        return None, reason
    if baseline == 0:
        return None, "zero_outside_error"
    return float(inside / baseline), None


def annotation_grids(sample, source_indices: Sequence[int],
                     target_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    """Build the canonical PhysLoc regions on an evaluator latent grid."""
    timeline = sample.timeline
    gate = lambda name: np.asarray(timeline[name], bool)[:, None, None]
    segmentation = np.asarray(sample.segmentations)
    twin_segmentation = (np.asarray(sample.twin.segmentations)
                         if sample.twin is not None else segmentation)
    foreground = (segmentation > 0) | (twin_segmentation > 0)
    raw = {
        "violating_object": np.asarray(sample.violator_mask, bool),
        "active_violation": np.asarray(sample.violation_mask, bool),
        "active_violation_visible": np.asarray(sample.visible_violation, bool),
        "expected_object": np.asarray(sample.reference_mask, bool) & gate("consequence"),
        "violating_object_active": (np.asarray(sample.violator_mask, bool)
                                     & gate("active")),
        "causal_consequence": (np.asarray(sample.causal) == 2) & gate("consequence"),
        "foreground": foreground,
    }
    grids = {name: project_mask(value, source_indices, target_shape)
             for name, value in raw.items()}
    grids["severity"] = project_volume(
        np.asarray(sample.severity_map, np.float32), source_indices,
        target_shape, "mean")
    grids["outside_violation_foreground"] = (
        grids["foreground"] & ~grids["active_violation"])
    grids["outside_violation_all"] = ~grids["active_violation"]
    return grids


def _metric(value: Optional[float], reason: Optional[str]) -> Dict[str, object]:
    return {"value": value, "available": reason is None, "reason": reason}


def localization_metrics(error: np.ndarray, grids: Dict[str, np.ndarray]
                         ) -> Dict[str, Dict[str, object]]:
    """Compute per-clip spatial and spatio-temporal localization metrics."""
    regions = ("violating_object", "active_violation",
               "active_violation_visible", "expected_object",
               "causal_consequence")
    spatial: Dict[str, object] = {}
    spatiotemporal: Dict[str, object] = {}
    for name in regions:
        target = grids[name]
        value, reason = masked_mean(error, target)
        weighted, weighted_reason = masked_mean(error, target, grids["severity"])
        spatial[name] = {
            "ppe": _metric(value, reason),
            "severity_weighted_ppe": _metric(weighted, weighted_reason),
            "tokens": int(target.sum()),
        }
        candidates = grids["foreground"] | target
        value, reason = average_precision(error, target, candidates)
        ratio, ratio_reason = error_ratio(
            error, target, grids["foreground"] & ~target)
        spatiotemporal[name + "_ap"] = _metric(value, reason)
        spatiotemporal[name + "_error_ratio"] = _metric(ratio, ratio_reason)
    for name in ("outside_violation_foreground", "outside_violation_all"):
        value, reason = masked_mean(error, grids[name])
        spatial[name] = {"ppe": _metric(value, reason),
                         "tokens": int(grids[name].sum())}
    return {"spatial": spatial, "spatiotemporal": spatiotemporal}


def temporal_metrics(error: np.ndarray, sample, source_indices: Sequence[int]
                     ) -> Dict[str, object]:
    """Per-frame PPE and temporal windows/AP at native latent resolution."""
    trace = np.asarray(error, np.float64).mean(axis=(1, 2))
    timeline = sample.timeline
    T = len(trace)
    clocks = {name: project_trace(np.asarray(timeline[name], bool), source_indices,
                                  T, "max") > 0
              for name in ("active", "observable", "occluded", "consequence")}
    positive = np.flatnonzero(clocks["active"] | clocks["consequence"])
    before = np.arange(T) < (int(positive[0]) if positive.size else 0)
    windows = {
        "before": before,
        "active_visible": clocks["active"] & clocks["observable"] & ~clocks["occluded"],
        "active_hidden": clocks["active"] & (clocks["occluded"] | ~clocks["observable"]),
        "consequence": clocks["consequence"],
    }
    aggregates = {}
    for name, mask in windows.items():
        value, reason = masked_mean(trace, mask)
        aggregates[name] = _metric(value, reason)
    event_ap = _metric(*average_precision(trace, clocks["active"]))
    consequence_ap = _metric(*average_precision(trace, clocks["consequence"]))
    return {
        "latent_frame_ppe": trace.tolist(),
        "rgb_frame_ppe": expand_latent_trace(trace, len(source_indices)).tolist(),
        "windows": aggregates,
        "temporal_event_ap": event_ap,
        "temporal_consequence_ap": consequence_ap,
        "latent_clocks": {name: values.astype(int).tolist()
                          for name, values in clocks.items()},
    }
