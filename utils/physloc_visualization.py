"""Per-clip and per-pair visual evidence for localized PhysLoc PPE."""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .physloc_metrics import temporal_bins, temporal_metrics


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _resize(frame: np.ndarray, width: int = 320) -> np.ndarray:
    height = max(1, int(round(frame.shape[0] * width / frame.shape[1])))
    return cv2.resize(np.asarray(frame), (width, height), interpolation=cv2.INTER_AREA)


def expand_error_grid(error: np.ndarray, sampled_frames: int) -> np.ndarray:
    value = np.asarray(error, np.float32)
    out = np.zeros((sampled_frames,) + value.shape[1:], np.float32)
    for t, frames in enumerate(temporal_bins(sampled_frames, len(value))):
        out[frames] = value[t]
    return out


def _outline(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel) > cv2.erode(mask, kernel)


def annotation_overlay(rgb: np.ndarray, active: Optional[np.ndarray] = None,
                       visible: Optional[np.ndarray] = None,
                       expected: Optional[np.ndarray] = None,
                       causal: Optional[np.ndarray] = None) -> np.ndarray:
    """Red=active, white=visible, green=lawful reference, blue=consequence."""
    image = np.asarray(rgb, np.uint8).copy()
    if active is not None:
        mask = np.asarray(active, bool)
        image[mask] = (0.55 * image[mask] + 0.45 * np.array([255, 55, 55])).astype(np.uint8)
    for mask, colour in ((visible, (255, 255, 255)),
                         (expected, (70, 235, 120)),
                         (causal, (80, 150, 255))):
        if mask is not None and np.asarray(mask, bool).any():
            image[_outline(mask)] = colour
    return image


def compose_frame(rgb: np.ndarray, error: np.ndarray, scale: float,
                  active=None, visible=None, expected=None, causal=None) -> np.ndarray:
    rgb = _resize(rgb)
    h, w = rgb.shape[:2]
    active = None if active is None else cv2.resize(
        np.asarray(active, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    visible = None if visible is None else cv2.resize(
        np.asarray(visible, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    expected = None if expected is None else cv2.resize(
        np.asarray(expected, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    causal = None if causal is None else cv2.resize(
        np.asarray(causal, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    annotation = annotation_overlay(rgb, active, visible, expected, causal)
    magnitude = cv2.resize(np.asarray(error, np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(magnitude / max(float(scale), 1e-12), 0, 1)
    heat = np.ascontiguousarray(cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1])
    overlay = (0.45 * rgb + 0.55 * heat).astype(np.uint8)
    panels = [(np.ascontiguousarray(rgb), "RGB"),
              (np.ascontiguousarray(annotation), "ANNOTATIONS"),
              (heat, "PPE ERROR"), (overlay, "PPE OVERLAY")]
    for panel, title in panels:
        cv2.rectangle(panel, (0, 0), (w, 25), (15, 15, 20), -1)
        cv2.putText(panel, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([panel for panel, _ in panels], axis=1)


def _raw_annotations(sample, indices: Sequence[int]):
    if sample is None or sample.is_valid:
        return None
    idx = np.clip(np.asarray(indices, int), 0, sample.num_frames - 1)
    consequence = np.asarray(sample.timeline["consequence"], bool)[:, None, None]
    return {
        "active": np.asarray(sample.violation_mask, bool)[idx],
        "visible": np.asarray(sample.visible_violation, bool)[idx],
        "expected": (np.asarray(sample.reference_mask, bool) & consequence)[idx],
        "causal": ((np.asarray(sample.causal) == 2) & consequence)[idx],
        "severity": np.asarray(sample.severity_map, np.float32)[idx],
    }


def render_clip(sample, error: np.ndarray, indices: Sequence[int], scale: float,
                output_mp4: str, output_png: str, annotation_sample=None) -> None:
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    source = np.asarray(sample.video)[np.clip(np.asarray(indices, int), 0,
                                              sample.num_frames - 1)]
    errors = expand_error_grid(error, len(source))
    annotations = _raw_annotations(annotation_sample, indices)
    frames = []
    for t, rgb in enumerate(source):
        kwargs = ({name: annotations[name][t] for name in
                   ("active", "visible", "expected", "causal")}
                  if annotations is not None else {})
        frames.append(compose_frame(rgb, errors[t], scale, **kwargs))
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, float(sample.fps)), (width, height))
    if not writer.isOpened():
        raise IOError("could not open video writer for %s" % output_mp4)
    for frame in frames:
        writer.write(frame[..., ::-1])
    writer.release()
    _clip_summary(sample, error, indices, scale, output_png, annotation_sample)


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/data/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/data/tmp/cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clip_summary(sample, error, indices, scale, output_png, annotation_sample):
    plt = _pyplot()
    frames = np.asarray(sample.video)[np.clip(np.asarray(indices, int), 0,
                                              sample.num_frames - 1)]
    expanded = expand_error_grid(error, len(frames))
    picks = np.unique(np.linspace(0, len(frames) - 1, min(4, len(frames))).round().astype(int))
    fig = plt.figure(figsize=(4 * len(picks), 9))
    grid = fig.add_gridspec(3, len(picks), height_ratios=(1, 1, .75))
    for column, t in enumerate(picks):
        rgb_ax, error_ax = fig.add_subplot(grid[0, column]), fig.add_subplot(grid[1, column])
        rgb_ax.imshow(frames[t])
        rgb_ax.set_title("sampled frame %d" % t)
        error_ax.imshow(frames[t])
        error_ax.imshow(cv2.resize(expanded[t],
                                   (frames[t].shape[1], frames[t].shape[0])),
                        cmap="inferno", alpha=.65, vmin=0, vmax=scale)
        error_ax.set_title("PPE magnitude")
        rgb_ax.axis("off")
        error_ax.axis("off")
    trace_ax = fig.add_subplot(grid[2, :])
    trace = expanded.mean(axis=(1, 2))
    trace_ax.plot(np.arange(len(trace)), trace, color="#d84a4a", label="PPE")
    if annotation_sample is not None:
        annotations = _raw_annotations(annotation_sample, indices)
        severity = annotations["severity"].reshape(len(trace), -1).max(axis=1)
        severity_ax = trace_ax.twinx()
        severity_ax.plot(np.arange(len(trace)), severity, color="#ef9b35",
                         alpha=.8, label="severity")
        severity_ax.set_ylabel("severity [0,1]", color="#ef9b35")
        for name, colour in (("active", "#ff7777"),
                             ("consequence", "#7698d8")):
            values = np.asarray(annotation_sample.timeline[name], bool)[
                np.clip(np.asarray(indices, int), 0, annotation_sample.num_frames - 1)]
            trace_ax.fill_between(np.arange(len(trace)), 0, 1, where=values,
                                  transform=trace_ax.get_xaxis_transform(),
                                  color=colour, alpha=.13, label=name)
    trace_ax.set(xlabel="sampled RGB frame", ylabel="PPE")
    trace_ax.grid(alpha=.25)
    trace_ax.legend(loc="upper left", ncol=3)
    fig.suptitle(str(sample.uid))
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def render_pair_summary(valid_runtime: Dict, invalid_runtime: Dict,
                        output_png: str) -> None:
    plt = _pyplot()
    sample = invalid_runtime["sample"]
    indices = invalid_runtime["indices"]
    valid = temporal_metrics(valid_runtime["grid"], sample, indices)
    invalid = temporal_metrics(invalid_runtime["grid"], sample, indices)
    x = np.arange(len(invalid["rgb_frame_ppe"]))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, valid["rgb_frame_ppe"], label="valid twin", color="#39a96b")
    ax.plot(x, invalid["rgb_frame_ppe"], label="invalid", color="#d84a4a")
    clocks = invalid["latent_clocks"]
    for label, colour in (("active", "#ff7777"), ("consequence", "#7698d8")):
        expanded = np.zeros(len(x), bool)
        for t, frames in enumerate(temporal_bins(len(x), len(clocks[label]))):
            expanded[frames] = bool(clocks[label][t])
        ax.fill_between(x, 0, 1, where=expanded, transform=ax.get_xaxis_transform(),
                        color=colour, alpha=.15, label=label)
    ax.set(title="Pair PPE trace: %s" % sample.uid,
           xlabel="sampled RGB frame", ylabel="PPE")
    ax.grid(alpha=.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def render_pair_artifacts(run_dir: str, pair, runtime: Dict[str, Dict],
                          model: str = "model") -> None:
    available = [entry for entry in runtime.values() if entry.get("grid") is not None]
    if not available:
        return
    pool = np.concatenate([entry["grid"].ravel() for entry in available])
    positive = pool[pool > 0]
    scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    clips_dir = os.path.join(run_dir, "visualizations", "clips", _safe(model))
    pairs_dir = os.path.join(run_dir, "visualizations", "pairs", _safe(model))
    valid = runtime.get(pair.valid.uid)
    if valid is not None and valid.get("grid") is not None:
        stem = _safe(pair.valid.uid)
        render_clip(pair.valid, valid["grid"], valid["indices"], scale,
                    os.path.join(clips_dir, stem + ".mp4"),
                    os.path.join(clips_dir, stem + ".png"))
    for invalid_sample in pair.invalids:
        invalid = runtime.get(invalid_sample.uid)
        if invalid is None or invalid.get("grid") is None:
            continue
        stem = _safe(invalid_sample.uid)
        render_clip(invalid_sample, invalid["grid"], invalid["indices"], scale,
                    os.path.join(clips_dir, stem + ".mp4"),
                    os.path.join(clips_dir, stem + ".png"), invalid_sample)
        if valid is not None and valid.get("grid") is not None:
            render_pair_summary(valid, invalid,
                                os.path.join(pairs_dir, stem + "_pair.png"))
