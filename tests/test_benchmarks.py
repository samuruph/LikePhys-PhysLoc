import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.common import compute_misrank_normalized
from benchmarks.likephys import evaluate
from benchmarks.physloc import _sample_difficulty, resolve_root


class CommonBenchmarkTests(unittest.TestCase):
    def test_misrank_is_shared_across_dataset_adapters(self):
        results = {
            "group": {
                "valid": {"v": {"loss": 2.0}},
                "invalid": {"i": {"loss": 1.0}},
            }
        }
        metric = compute_misrank_normalized(results)["invalid"]
        self.assertEqual(metric["misrank_ratio"], 1.0)
        self.assertEqual(metric["total_pairs"], 1)

    def test_ties_preserve_legacy_non_misrank_semantics(self):
        results = {
            "group": {
                "valid": {"v": {"loss": 1.0}},
                "invalid": {"i": {"loss": 1.0}},
            }
        }
        self.assertEqual(
            compute_misrank_normalized(results)["invalid"]["misrank_ratio"],
            0.0,
        )


class DatasetAdapterTests(unittest.TestCase):
    def test_difficulty_falls_back_to_canonical_scene_info(self):
        legacy_loader_sample = SimpleNamespace(scene_info={
            "difficulty_analysis": {"level": "hard"},
        })
        self.assertEqual(_sample_difficulty(legacy_loader_sample)["level"], "hard")

    def test_likephys_adapter_groups_sorted_mp4_files(self):
        calls = []

        def score(args, path, pipe):
            calls.append(os.path.basename(path))
            return 1.5, {
                "noise_pred_mean": 0.0,
                "true_noise_mean": 0.0,
                "loss_array": [1.5],
            }

        with tempfile.TemporaryDirectory() as root:
            subgroup = os.path.join(root, "scene")
            os.makedirs(subgroup)
            for name in ("valid_00.mp4", "fall_up_00.mp4", "notes.txt"):
                Path(subgroup, name).touch()
            args = SimpleNamespace(seed=4)
            result = evaluate(args, root, object(), score)

        self.assertEqual(calls, ["fall_up_00.mp4", "valid_00.mp4"])
        self.assertIn("fall_up", result["scene"])
        self.assertEqual(args.subgroup_seed, 4)

    def test_physloc_root_is_validated_before_model_loading(self):
        with tempfile.TemporaryDirectory() as root:
            sample = Path(root, "samples", "release", "pair", "valid")
            sample.mkdir(parents=True)
            Path(sample, "sample.json").write_text("{}", encoding="utf-8")
            Path(sample, "data.h5").touch()
            args = SimpleNamespace(
                physloc_root=root,
                physloc_hub_repo=None,
                physloc_cache="unused",
                physloc_loader=None,
            )
            self.assertEqual(resolve_root(args), root)
            self.assertTrue(args.physloc_loader.endswith("physloc/loader.py"))
            self.assertIsNone(args.physloc_prompt)


if __name__ == "__main__":
    unittest.main()
