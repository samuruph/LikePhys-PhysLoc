"""Per-clip and per-pair visual evidence for localized denoising error."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .physloc_metrics import temporal_bins, temporal_metrics


def _diverging_lut() -> np.ndarray:
    """Build a 256-entry BGR lookup table from matplotlib's coolwarm colormap.

    OpenCV's COLORMAP_JET is a rainbow map: it isn't perceptually uniform and
    introduces false banding/edges in the middle of the range, which is
    exactly where a signed invalid-minus-valid delta needs to be readable.
    coolwarm is diverging, perceptually ordered, and white at zero.
    """
    import matplotlib
    colormap = matplotlib.colormaps["coolwarm"]
    colors = colormap(np.linspace(0, 1, 256))[:, :3] * 255
    return np.ascontiguousarray(colors.astype(np.uint8))  # RGB, 256x3


_DIVERGING_LUT = _diverging_lut()


def _safe(value: object) -> str:
    """Convert an identifier to a safe filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _resize(frame: np.ndarray, width: int = 320) -> np.ndarray:
    """Resize an image to a fixed width while preserving aspect ratio."""
    height = max(1, int(round(frame.shape[0] * width / frame.shape[1])))
    return cv2.resize(np.asarray(frame), (width, height), interpolation=cv2.INTER_AREA)


def expand_error_grid(error: np.ndarray, sampled_frames: int) -> np.ndarray:
    """Expand latent-frame error grids onto sampled RGB-frame bins."""
    value = np.asarray(error, np.float32)
    out = np.zeros((sampled_frames,) + value.shape[1:], np.float32)
    for t, frames in enumerate(temporal_bins(sampled_frames, len(value))):
        out[frames] = value[t]
    return out


def _outline(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel morphological outline for a binary mask."""
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


def _heatmap(error: np.ndarray, width: int, height: int, scale: float,
             signed: bool = False) -> np.ndarray:
    """Render an absolute or signed error map at display resolution."""
    magnitude = cv2.resize(np.asarray(error, np.float32), (width, height),
                           interpolation=cv2.INTER_LINEAR)
    if signed:
        normalized = np.clip(magnitude / max(float(scale), 1e-12), -1, 1)
        indexed = ((normalized + 1.0) * 127.5).astype(np.uint8)
        return np.ascontiguousarray(_DIVERGING_LUT[indexed])
    normalized = np.clip(magnitude / max(float(scale), 1e-12), 0, 1)
    return np.ascontiguousarray(cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1])


def _label_panel(panel: np.ndarray, title: str) -> np.ndarray:
    """Add a consistent title strip to one visualization panel."""
    panel = np.asarray(panel, np.uint8).copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (15, 15, 20), -1)
    cv2.putText(panel, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return panel


def _draw_legend(panel: np.ndarray, y0: int, block_h: int, width: int,
                 scale: float, signed: bool, title: str) -> None:
    """Draw one compact color-scale legend into an existing panel in place."""
    x0, x1 = 14, width - 14
    cv2.putText(panel, title, (10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.32, (195, 195, 205), 1, cv2.LINE_AA)
    bar_y0 = y0 + 16
    bar_y1 = max(bar_y0 + 4, y0 + block_h - 16)
    values = np.linspace(-1 if signed else 0, 1,
                         max(2, x1 - x0), dtype=np.float32)[None, :]
    if signed:
        indexed = ((values + 1.0) * 127.5).astype(np.uint8)
        colors = _DIVERGING_LUT[indexed][0]
        ticks = [(x0, "-%.3f" % scale), ((x0 + x1) // 2, "0"),
                (x1 - 1, "+%.3f" % scale)]
    else:
        indexed = (np.clip(values, 0, 1) * 255).astype(np.uint8)
        colors = cv2.applyColorMap(indexed, cv2.COLORMAP_INFERNO)[0, :, ::-1]
        ticks = [(x0, "0"), (x1 - 1, "%.3f" % scale)]
    bar = np.repeat(colors[None, :, :], bar_y1 - bar_y0, axis=0)
    panel[bar_y0:bar_y1, x0:x1] = bar
    for position, label in ticks:
        cv2.line(panel, (position, bar_y1), (position, bar_y1 + 4),
                 (230, 230, 230), 1)
        text_x = max(2, min(width - 60, position - 20))
        cv2.putText(panel, label, (text_x, min(y0 + block_h - 2, bar_y1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1,
                    cv2.LINE_AA)


def _info_panel(width: int, height: int, frame_index: int, valid_loss: float,
                invalid_loss: float, raw_scale: float,
                difference_scale: float) -> np.ndarray:
    """Combine per-frame metrics with the absolute and delta color legends."""
    panel = np.zeros((height, width, 3), np.uint8)
    panel[:] = (32, 32, 38)
    panel = _label_panel(panel, "FRAME INFO")
    cv2.putText(panel, "frame %d" % frame_index, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel,
                "valid %.4f  invalid %.4f  delta %+.4f"
                % (valid_loss, invalid_loss, invalid_loss - valid_loss),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240),
                1, cv2.LINE_AA)
    block = max(40, (height - 68) // 2)
    _draw_legend(panel, 68, block, width, raw_scale, False, "ABSOLUTE ERROR")
    _draw_legend(panel, 68 + block, block, width, difference_scale, True,
                "SIGNED DELTA")
    return panel


def _resolve_ffmpeg() -> str:
    """Find ffmpeg even when the evaluator was launched with a short PATH."""
    candidates = [
        os.environ.get("FFMPEG_BINARY"),
        shutil.which("ffmpeg"),
        "/home/ec2-user/miniconda3/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise IOError(
        "ffmpeg encoder not found; set FFMPEG_BINARY or add ffmpeg to PATH")


def compose_frame(rgb: np.ndarray, error: np.ndarray, scale: float,
                  active=None, visible=None, expected=None, causal=None) -> np.ndarray:
    """Compose RGB, annotation, denoising-error heatmap, and overlay panels."""
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
    heat = _heatmap(error, w, h, scale)
    overlay = (0.45 * rgb + 0.55 * heat).astype(np.uint8)
    panels = [_label_panel(rgb, "RGB"),
              _label_panel(annotation, "ANNOTATIONS"),
              _label_panel(heat, "DENOISING ERROR"),
              _label_panel(overlay, "ERROR OVERLAY")]
    return np.concatenate(panels, axis=1)


def compose_pair_frame(valid_rgb: np.ndarray, invalid_rgb: np.ndarray,
                       invalid_error: np.ndarray, valid_error: np.ndarray,
                       difference: np.ndarray, raw_scale: float,
                       difference_scale: float, frame_index: int,
                       invalid_loss: float, valid_loss: float) -> np.ndarray:
    """Compose one synchronized valid/invalid pair frame as a 3x3 grid.

    The delta panel is the signed invalid-minus-valid residual. Red means the
    invalid clip has higher error; blue means the valid clip has higher error.
    """
    invalid_rgb = _resize(invalid_rgb)
    valid_rgb = _resize(valid_rgb)
    height, width = invalid_rgb.shape[:2]
    invalid_heat = _heatmap(invalid_error, width, height, raw_scale)
    valid_heat = _heatmap(valid_error, width, height, raw_scale)
    difference_heat = _heatmap(
        difference, width, height, difference_scale, signed=True)
    valid_overlay = (0.45 * valid_rgb + 0.55 * valid_heat).astype(np.uint8)
    invalid_overlay = (0.45 * invalid_rgb + 0.55 * invalid_heat).astype(np.uint8)
    difference_overlay = (0.45 * invalid_rgb + 0.55 * difference_heat).astype(np.uint8)
    info = _info_panel(width, height, frame_index, valid_loss, invalid_loss,
                       raw_scale, difference_scale)
    rows = [
        [
            _label_panel(valid_rgb, "VALID RGB"),
            _label_panel(valid_heat, "VALID ERROR"),
            _label_panel(valid_overlay, "VALID OVERLAY"),
        ],
        [
            _label_panel(invalid_rgb, "INVALID RGB"),
            _label_panel(invalid_heat, "INVALID ERROR"),
            _label_panel(invalid_overlay, "INVALID OVERLAY"),
        ],
        [
            _label_panel(difference_heat, "DELTA: INVALID - VALID"),
            _label_panel(difference_overlay, "DELTA OVERLAY"),
            info,
        ],
    ]
    return np.vstack([np.hstack(row) for row in rows])


def _raw_annotations(sample, indices: Sequence[int]):
    """Load source-resolution masks for selected RGB frames."""
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
    """Write one synchronized clip video and representative-frame summary."""
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
    video_frames = np.stack(frames)
    pad_h = (-video_frames.shape[1]) % 16
    pad_w = (-video_frames.shape[2]) % 16
    if pad_h or pad_w:
        video_frames = np.pad(
            video_frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="edge")
    height, width = video_frames.shape[1:3]
    ffmpeg = _resolve_ffmpeg()
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "%dx%d" % (width, height),
        "-r", str(max(1.0, float(sample.fps))), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_mp4,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in video_frames:
            process.stdin.write(np.ascontiguousarray(frame[..., ::-1]).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise IOError("ffmpeg could not encode H.264 video: %s" % output_mp4)
    except Exception:
        process.kill()
        process.wait()
        raise
    _clip_summary(sample, error, indices, scale, output_png, annotation_sample)


def render_pair_clip(valid_runtime: Dict, invalid_runtime: Dict,
                     raw_scale: float, difference_scale: float,
                     output_mp4: str) -> None:
    """Write one combined valid/invalid/difference visualization video."""
    valid_sample = valid_runtime["sample"]
    invalid_sample = invalid_runtime["sample"]
    indices = np.asarray(invalid_runtime["indices"], int)
    valid_source = np.asarray(valid_sample.video)[np.clip(
        indices, 0, valid_sample.num_frames - 1)]
    invalid_source = np.asarray(invalid_sample.video)[np.clip(
        indices, 0, invalid_sample.num_frames - 1)]
    valid_errors = expand_error_grid(valid_runtime["grid"], len(indices))
    invalid_errors = expand_error_grid(invalid_runtime["grid"], len(indices))
    differences = invalid_errors - valid_errors
    valid_loss = float(valid_runtime["info"]["loss"])
    invalid_loss = float(invalid_runtime["info"]["loss"])
    frames = [
        compose_pair_frame(
            valid_source[t], invalid_source[t], invalid_errors[t],
            valid_errors[t], differences[t], raw_scale, difference_scale,
            t, invalid_loss, valid_loss)
        for t in range(len(indices))
    ]
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    video_frames = np.stack(frames)
    pad_h = (-video_frames.shape[1]) % 16
    pad_w = (-video_frames.shape[2]) % 16
    if pad_h or pad_w:
        video_frames = np.pad(
            video_frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="edge")
    height, width = video_frames.shape[1:3]
    ffmpeg = _resolve_ffmpeg()
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "%dx%d" % (width, height),
        "-r", str(max(1.0, float(invalid_sample.fps))), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_mp4,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in video_frames:
            process.stdin.write(np.ascontiguousarray(frame[..., ::-1]).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise IOError("ffmpeg could not encode H.264 video: %s" % output_mp4)
    except Exception:
        process.kill()
        process.wait()
        raise


def _pyplot():
    """Load matplotlib with a headless backend and writable cache paths."""
    os.environ.setdefault("MPLCONFIGDIR", "/data/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/data/tmp/cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clip_summary(sample, error, indices, scale, output_png, annotation_sample):
    """Write representative frames and synchronized error/severity traces."""
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
        error_ax.set_title("Mean squared denoising error")
        rgb_ax.axis("off")
        error_ax.axis("off")
    trace_ax = fig.add_subplot(grid[2, :])
    trace = expanded.mean(axis=(1, 2))
    trace_ax.plot(np.arange(len(trace)), trace, color="#d84a4a",
                  label="mean squared denoising error")
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
    trace_ax.set(xlabel="sampled RGB frame",
                 ylabel="mean squared denoising error")
    trace_ax.grid(alpha=.25)
    trace_ax.legend(loc="upper left", ncol=3)
    fig.suptitle(str(sample.uid))
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def render_pair_summary(valid_runtime: Dict, invalid_runtime: Dict,
                        output_png: str) -> None:
    """Plot valid and invalid denoising-error traces on shared axes."""
    plt = _pyplot()
    sample = invalid_runtime["sample"]
    indices = invalid_runtime["indices"]
    valid = temporal_metrics(valid_runtime["grid"], sample, indices)
    invalid = temporal_metrics(invalid_runtime["grid"], sample, indices)
    x = np.arange(len(invalid["rgb_frame_ppe"]))
    difference = (np.asarray(invalid["rgb_frame_ppe"], np.float64)
                  - np.asarray(valid["rgb_frame_ppe"], np.float64))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, valid["rgb_frame_ppe"], label="valid denoising error",
            color="#39a96b")
    ax.plot(x, invalid["rgb_frame_ppe"], label="invalid denoising error",
            color="#d84a4a")
    difference_ax = ax.twinx()
    difference_ax.plot(x, difference, label="invalid - valid",
                       color="#6f42c1", linestyle="--", alpha=.9)
    difference_ax.axhline(0.0, color="#6f42c1", linewidth=.7, alpha=.35)
    difference_ax.set_ylabel("invalid - valid PPE", color="#6f42c1")
    clocks = invalid["latent_clocks"]
    for label, colour in (("active", "#ff7777"), ("consequence", "#7698d8")):
        expanded = np.zeros(len(x), bool)
        for t, frames in enumerate(temporal_bins(len(x), len(clocks[label]))):
            expanded[frames] = bool(clocks[label][t])
        ax.fill_between(x, 0, 1, where=expanded, transform=ax.get_xaxis_transform(),
                        color=colour, alpha=.15, label=label)
    ax.set(title="Pair denoising-error trace: %s | valid PPE %.4f, invalid PPE %.4f"
           % (sample.uid, valid_runtime["info"]["loss"],
              invalid_runtime["info"]["loss"]),
           xlabel="sampled RGB frame",
            ylabel="mean squared denoising error")
    ax.grid(alpha=.25)
    handles, labels = ax.get_legend_handles_labels()
    delta_handles, delta_labels = difference_ax.get_legend_handles_labels()
    ax.legend(handles + delta_handles, labels + delta_labels, ncol=5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def render_pair_artifacts(run_dir: str, pair, runtime: Dict[str, Dict],
                          model: str = "model") -> None:
    """Render one combined artifact set for every valid/invalid pair."""
    available = [entry for entry in runtime.values() if entry.get("grid") is not None]
    if not available:
        return
    pool = np.concatenate([entry["grid"].ravel() for entry in available])
    positive = pool[pool > 0]
    raw_scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    valid = runtime.get(pair.valid.uid)
    if valid is None or valid.get("grid") is None:
        return
    differences = [
        runtime[sample.uid]["grid"] - valid["grid"]
        for sample in pair.invalids
        if runtime.get(sample.uid, {}).get("grid") is not None
    ]
    delta_values = np.concatenate([value.ravel() for value in differences]) \
        if differences else np.asarray([], np.float32)
    difference_scale = (float(np.percentile(np.abs(delta_values), 99.0))
                        if delta_values.size else 1.0)
    output_dir = os.path.join(run_dir, "visualizations", _safe(model))
    for invalid_sample in pair.invalids:
        invalid = runtime.get(invalid_sample.uid)
        if invalid is None or invalid.get("grid") is None:
            continue
        stem = _safe(invalid_sample.uid)
        render_pair_clip(
            valid, invalid, raw_scale, difference_scale,
            os.path.join(output_dir, stem + ".mp4"))
        render_pair_summary(
            valid, invalid, os.path.join(output_dir, stem + ".png"))
