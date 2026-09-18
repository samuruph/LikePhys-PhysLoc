"""LikePhys scenario configuration and dataset traversal."""
from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

from tqdm.auto import tqdm

from ..datasets.likephys import DATASETS, PROMPTS, iter_video_groups

ScoreVideo = Callable[[object, str, object], Tuple[object, Dict[str, object]]]


def evaluate(args: object, dataset_dir: str, pipe: object,
             score_video: ScoreVideo) -> Dict[str, object]:
    """Evaluate LikePhys MP4 files grouped by their scenario subdirectory."""
    results: Dict[str, object] = {}
    groups = list(iter_video_groups(dataset_dir))
    for index, (subgroup_id, videos) in tqdm(
            enumerate(groups), total=len(groups),
            desc="Evaluating LikePhys subgroups"):
        args.subgroup_seed = args.seed + index
        subgroup_results: Dict[str, object] = {}
        for video_name in videos:
            video_path = os.path.join(dataset_dir, subgroup_id, video_name)
            variation = video_name.rsplit("_", 1)[0]
            loss, log_info = score_video(args, video_path, pipe)
            if loss is None:
                continue
            subgroup_results.setdefault(variation, {})[video_name] = {
                "loss": loss,
                "noise_pred_mean": log_info["noise_pred_mean"],
                "true_noise_mean": log_info["true_noise_mean"],
                "loss_array": log_info["loss_array"],
            }
            if "evaluation_time_seconds" in log_info:
                subgroup_results[variation][video_name][
                    "evaluation_time_seconds"] = log_info["evaluation_time_seconds"]
        if subgroup_results:
            results[subgroup_id] = subgroup_results
    return results
