import unittest

import numpy as np

from utils.physloc_metrics import (
    annotation_grids, average_precision, error_ratio, expand_latent_trace,
    localization_metrics, masked_mean, parse_score_groups, project_mask,
    project_volume, temporal_bins, temporal_metrics, normalize_condition,
)


class ProjectionTests(unittest.TestCase):
    def test_four_to_one_temporal_bins_preserve_first_frame(self):
        bins = temporal_bins(9, 3)
        self.assertEqual([x.tolist() for x in bins], [[0], [1, 2, 3, 4], [5, 6, 7, 8]])

    def test_project_mask_keeps_small_regions(self):
        mask = np.zeros((5, 8, 8), bool)
        mask[2, 1, 1] = True
        projected = project_mask(mask, range(5), (3, 2, 2))
        self.assertTrue(projected.any())

    def test_projection_mean_and_trace_expansion(self):
        value = np.arange(5, dtype=float)[:, None, None] * np.ones((5, 2, 2))
        out = project_volume(value, range(5), (2, 1, 1), "mean")
        self.assertEqual(out.shape, (2, 1, 1))
        np.testing.assert_allclose(expand_latent_trace([2, 8], 5), [2, 8, 8, 8, 8])


class MetricTests(unittest.TestCase):
    def test_parse_score_groups(self):
        self.assertEqual(parse_score_groups(["base_ppe,temporal_ppe"]),
                         ("base_ppe", "temporal_ppe"))
        self.assertEqual(set(parse_score_groups(["all"])),
                         {"base_ppe", "temporal_ppe", "spatial_ppe", "spatiotemporal_ppe"})
        self.assertEqual(normalize_condition("camera+multi"), "multi_motion")
        with self.assertRaises(ValueError):
            parse_score_groups(["all,not_a_score"])

    def test_temporal_bins_reject_empty_latent_bins(self):
        with self.assertRaisesRegex(ValueError, "non-empty latent bins"):
            temporal_bins(2, 3)

    def test_masked_mean_and_weighted_mean(self):
        error = np.array([1.0, 3.0])
        mask = np.array([True, True])
        self.assertEqual(masked_mean(error, mask)[0], 2.0)
        self.assertEqual(masked_mean(error, mask, np.array([0.0, 1.0]))[0], 3.0)
        self.assertEqual(
            masked_mean(np.array([np.nan]), np.array([True]))[1],
            "non_finite_error",
        )

    def test_ap_and_ratio(self):
        error = np.array([0.1, 0.9, 0.2, 0.8])
        target = np.array([False, True, False, True])
        self.assertEqual(average_precision(error, target)[0], 1.0)
        self.assertAlmostEqual(error_ratio(error, target, ~target)[0], 0.85 / 0.15)

    def test_ap_is_invariant_to_tied_token_order(self):
        scores = np.array([1.0, 1.0])
        first = average_precision(scores, np.array([True, False]))[0]
        second = average_precision(scores, np.array([False, True]))[0]
        self.assertEqual(first, 0.5)
        self.assertEqual(first, second)

    def test_empty_metrics_are_explicit(self):
        score, reason = average_precision(np.ones(3), np.zeros(3, bool))
        self.assertIsNone(score)
        self.assertEqual(reason, "no_positive_tokens")

    def test_counterfactual_reference_is_gated_by_consequence(self):
        class Fake:
            pass
        sample = Fake()
        sample.timeline = {name: np.array([0, 1, 1, 0], bool) for name in
                           ("active", "observable", "consequence")}
        sample.timeline["occluded"] = np.array([0, 1, 1, 0], bool)
        sample.segmentations = np.zeros((4, 2, 2), np.uint16)
        sample.twin = Fake()
        sample.twin.segmentations = np.zeros_like(sample.segmentations)
        sample.twin.segmentations[:, 0, 0] = 2
        sample.violator_mask = np.zeros_like(sample.segmentations, bool)
        sample.violation_mask = np.zeros_like(sample.segmentations, bool)
        sample.violation_mask[1, 0, 0] = True
        sample.visible_violation = np.zeros_like(sample.segmentations, bool)
        sample.reference_mask = sample.twin.segmentations == 2
        sample.causal = np.zeros_like(sample.segmentations, np.uint8)
        sample.severity_map = sample.violation_mask.astype(np.float32)
        grids = annotation_grids(sample, range(4), (4, 2, 2))
        self.assertTrue(grids["expected_object"][1, 0, 0])
        self.assertTrue(grids["expected_object"][2, 0, 0])
        self.assertFalse(grids["expected_object"][0, 0, 0])

        error = np.zeros((4, 2, 2), float)
        error[1, 0, 0] = 2.0
        result = localization_metrics(error, grids)
        self.assertEqual(result["spatial"]["active_violation"]["ppe"]["value"], 2.0)
        self.assertIn("active_violating_object", result["spatial"])
        self.assertIn("active_violating_object_ap", result["spatiotemporal"])

    def test_temporal_metrics_keep_native_and_projected_traces(self):
        class Fake:
            timeline = {
                "active": np.array([0, 1, 1, 0], bool),
                "observable": np.array([0, 1, 1, 0], bool),
                "occluded": np.zeros(4, bool),
                "consequence": np.array([0, 1, 1, 0], bool),
            }
        error = np.arange(16, dtype=float).reshape(4, 2, 2)
        result = temporal_metrics(error, Fake(), range(4))
        self.assertEqual(len(result["latent_frame_ppe"]), 4)
        self.assertEqual(len(result["rgb_frame_ppe"]), 4)
        self.assertTrue(result["windows"]["active_visible"]["available"])


if __name__ == "__main__":
    unittest.main()
