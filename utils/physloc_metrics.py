"""Pure NumPy helpers for PhysLoc PPE localization and aggregation."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCORE_GROUPS = {"base_ppe", "temporal_ppe", "spatial_ppe",
                "spatiotemporal_ppe"}


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
