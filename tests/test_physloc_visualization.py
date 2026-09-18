import unittest

import numpy as np

from utils.physloc_visualization import (
    annotation_overlay, compose_frame, expand_error_grid,
)


class VisualizationTests(unittest.TestCase):
    def test_expand_grid_uses_temporal_bins(self):
        error = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
        out = expand_error_grid(error, 5)
        self.assertEqual(out.shape, (5, 2, 2))
        self.assertEqual(float(out[0].max()), 0.0)
        self.assertEqual(float(out[-1].min()), 1.0)

    def test_annotation_colours_are_distinct(self):
        rgb = np.zeros((8, 8, 3), np.uint8)
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        out = annotation_overlay(rgb, active=mask, expected=mask, causal=mask)
        self.assertTrue(out.any())

    def test_four_panel_composition(self):
        rgb = np.full((12, 16, 3), 100, np.uint8)
        error = np.ones((3, 4), np.float32)
        frame = compose_frame(rgb, error, 1.0)
        self.assertEqual(frame.shape[1], 4 * 320)
        self.assertEqual(frame.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
