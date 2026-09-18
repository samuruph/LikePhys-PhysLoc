"""Shared result aggregation for LikePhys and PhysLoc."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def compute_misrank_normalized(
        results: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]]
        ) -> Dict[str, Dict[str, object]]:
    """Compute mean subgroup mis-rank for every discovered invalid type.

    A pair is mis-ranked when its valid loss is strictly greater than its
    invalid loss. Ties therefore remain non-detections but are not counted as
    mis-ranks, preserving the repository's legacy metric semantics.
    """
    invalid_types = sorted({
        variation
        for subgroup in results.values()
        for variation in subgroup
        if variation != "valid"
    })
    print("Discovered invalid types: %s" % invalid_types)

    output: Dict[str, Dict[str, object]] = {}
    for variation in invalid_types:
        subgroup_misranks = []
        total_pairs = 0
        for subgroup in results.values():
            if "valid" not in subgroup or variation not in subgroup:
                continue
            valid_losses = [float(item["loss"])
                            for item in subgroup["valid"].values()]
            invalid_losses = [float(item["loss"])
                              for item in subgroup[variation].values()]
            pairs = [(valid, invalid)
                     for valid in valid_losses for invalid in invalid_losses]
            if not pairs:
                continue
            subgroup_misranks.append(
                sum(valid > invalid for valid, invalid in pairs) / len(pairs))
            total_pairs += len(pairs)
        output[variation] = {
            "misrank_ratio": (float(np.mean(subgroup_misranks))
                              if subgroup_misranks else 0.0),
            "total_pairs": total_pairs,
            "subgroup_misranks": subgroup_misranks,
        }
    return output
