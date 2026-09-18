import unittest

import numpy as np

from utils.physloc_metrics import (
    average_precision, error_ratio, expand_latent_trace, masked_mean,
    parse_score_groups, project_mask, project_volume, temporal_bins,
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

    def test_masked_mean_and_weighted_mean(self):
        error = np.array([1.0, 3.0])
        mask = np.array([True, True])
        self.assertEqual(masked_mean(error, mask)[0], 2.0)
        self.assertEqual(masked_mean(error, mask, np.array([0.0, 1.0]))[0], 3.0)

    def test_ap_and_ratio(self):
        error = np.array([0.1, 0.9, 0.2, 0.8])
        target = np.array([False, True, False, True])
        self.assertEqual(average_precision(error, target)[0], 1.0)
        self.assertAlmostEqual(error_ratio(error, target, ~target)[0], 0.85 / 0.15)

    def test_empty_metrics_are_explicit(self):
        score, reason = average_precision(np.ones(3), np.zeros(3, bool))
        self.assertIsNone(score)
        self.assertEqual(reason, "no_positive_tokens")


if __name__ == "__main__":
    unittest.main()
