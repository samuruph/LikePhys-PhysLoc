"""LikePhys scenario configuration and dataset traversal."""
from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

from tqdm.auto import tqdm


PROMPTS = {
    "ball_drop": "ball dropping and colliding with the ground, in empty background",
    "ball_collision": "two balls colliding with each other",
    "pendulum": "a pendulum swinging",
    "block_slide": "a block sliding on a slope",
    "fluid": "a droplet falling",
    "faucet": "fluid flowing from a faucet",
    "cloth": "a piece of cloth dropping to the obstacle on the ground",
    "flag": "a piece of cloth waving in the wind",
    "river": "fluid flowing in a tank with obstacles",
    "shadow": "light source moving around an object showing its shadow",
    "pyramid": "a cube crash into a pile of spheres",
    "shadowm": "camera moving around an object",
    "sample": "two balls colliding with each other",
}

DATASETS = {
    "ball_drop": {"dataset_dir": "./data/likephys/ball_drop_videos", "data_name": "ball_drop"},
    "ball_collision": {"dataset_dir": "./data/likephys/ball_collision_videos", "data_name": "ball_collision"},
    "pendulum": {"dataset_dir": "./data/likephys/pendulum_videos", "data_name": "pendulum"},
    "block_slide": {"dataset_dir": "./data/likephys/block_slide_videos", "data_name": "block_slide"},
    "fluid": {"dataset_dir": "./data/likephys/fluid_videos", "data_name": "fluid"},
    "faucet": {"dataset_dir": "./data/likephys/faucet_videos", "data_name": "faucet"},
    "cloth": {"dataset_dir": "./data/cloth_drape_videos", "data_name": "cloth"},
    "flag": {"dataset_dir": "./data/flag_videos", "data_name": "flag"},
    "river": {"dataset_dir": "./data/likephys/river_videos", "data_name": "river"},
    "shadow": {"dataset_dir": "./data/likephys/shadow_videos", "data_name": "shadow"},
    "pyramid": {"dataset_dir": "./data/likephys/pyramid_videos", "data_name": "pyramid"},
    "shadowm": {"dataset_dir": "./data/likephys/shadow_camera_videos", "data_name": "shadowm"},
    "sample": {"dataset_dir": "./data/likephys/sample_videos", "data_name": "sample"},
}

ScoreVideo = Callable[[object, str, object], Tuple[object, Dict[str, object]]]


def evaluate(args: object, dataset_dir: str, pipe: object,
             score_video: ScoreVideo) -> Dict[str, object]:
    """Evaluate LikePhys MP4 files grouped by their scenario subdirectory."""
    results: Dict[str, object] = {}
    subgroups = sorted(os.listdir(dataset_dir))
    for index, subgroup_id in tqdm(
            enumerate(subgroups), total=len(subgroups),
            desc="Evaluating LikePhys subgroups"):
        subgroup_path = os.path.join(dataset_dir, subgroup_id)
        if not os.path.isdir(subgroup_path):
            continue
        args.subgroup_seed = args.seed + index
        subgroup_results: Dict[str, object] = {}
        for video_name in sorted(os.listdir(subgroup_path)):
            if not video_name.endswith(".mp4"):
                continue
            video_path = os.path.join(subgroup_path, video_name)
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
        if subgroup_results:
            results[subgroup_id] = subgroup_results
    return results
