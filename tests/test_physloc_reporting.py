import tempfile
import unittest
from unittest import mock

from utils.physloc_reporting import (
    category_summaries, severity_sensitivity, tidy_rows,
    write_analysis_bundle, write_csv,
)


def _info(severity, loss, valid=1.0):
    return {
        "loss": loss,
        "taxonomy": {"family": "permanence", "severity": severity,
                     "complexity": "L0", "condition": "standard",
                     "difficulty": "easy"},
        "pair_metrics": {"base_ppe": {
            "valid_ppe": valid, "invalid_ppe": loss,
            "ppe_gap": loss - valid, "detected": loss > valid, "reason": None}},
    }


class ReportingTests(unittest.TestCase):
    def test_complete_severity_ladder(self):
        group = {"valid": {"v": {"loss": 1.0}}}
        for severity, loss in (("weak", 1.1), ("medium", 1.3), ("strong", 1.8)):
            group["permanence_" + severity] = {severity: _info(severity, loss)}
        rows = tidy_rows({"pair": group})
        result = severity_sensitivity(rows)
        score = result["by_score"][0]
        self.assertEqual(score["full_ladder"]["accuracy"], 1.0)
        self.assertEqual(score["ordered_pair_accuracy"]["strong>weak"]["accuracy"], 1.0)

    def test_incomplete_ladder_is_not_counted_as_full(self):
        group = {"valid": {"v": {"loss": 1.0}},
                 "permanence_weak": {"w": _info("weak", 1.1)},
                 "permanence_strong": {"s": _info("strong", 1.5)}}
        result = severity_sensitivity(tidy_rows({"pair": group}))["by_score"][0]
        self.assertEqual(result["full_ladder"]["count"], 0)
        self.assertEqual(result["ordered_pair_accuracy"]["strong>weak"]["count"], 1)

    def test_duplicate_severity_replicates_are_averaged(self):
        group = {
            "valid": {"v": {"loss": 1.0}},
            "weak_a": {"a": _info("weak", 1.1)},
            "weak_b": {"b": _info("weak", 1.3)},
            "medium": {"m": _info("medium", 1.4)},
            "strong": {"s": _info("strong", 1.8)},
        }
        matched = severity_sensitivity(tidy_rows({"pair": group}))["matched_groups"][0]
        self.assertAlmostEqual(matched["gaps"]["weak"], 0.2)
        self.assertEqual(matched["replicate_counts"]["weak"], 2)
        self.assertTrue(matched["full_ladder"])

    def test_categories_and_csv(self):
        rows = tidy_rows({"pair": {"valid": {"v": {"loss": 1.0}},
                                    "x": {"i": _info("strong", 1.5)}}})
        summaries = category_summaries(rows)
        self.assertTrue(any(row["dimension"] == "severity" for row in summaries))
        with tempfile.TemporaryDirectory() as root:
            path = root + "/rows.csv"
            write_csv(path, rows)
            with open(path, encoding="utf-8") as handle:
                self.assertTrue(handle.read().startswith("pair_uid"))

    def test_analysis_bundle_writes_data_and_plots(self):
        group = {"valid": {"v": {"loss": 1.0}}}
        for severity, loss in (("weak", 1.1), ("medium", 1.3), ("strong", 1.8)):
            group["permanence_" + severity] = {severity: _info(severity, loss)}
        rows = tidy_rows({"pair": group})
        categories = category_summaries(rows)
        severity = severity_sensitivity(rows)
        with tempfile.TemporaryDirectory() as root:
            write_analysis_bundle(root, "model", rows, categories, severity)
            self.assertTrue(__import__("os").path.exists(
                root + "/analysis/data/category_summary_model.csv"))
            self.assertTrue(__import__("os").path.exists(
                root + "/analysis/plots/model/severity_trends.png"))

    def test_plot_failure_does_not_discard_csv_outputs(self):
        rows = tidy_rows({"pair": {
            "valid": {"v": {"loss": 1.0}},
            "x": {"i": _info("strong", 1.5)},
        }})
        categories = category_summaries(rows)
        severity = severity_sensitivity(rows)
        with tempfile.TemporaryDirectory() as root, mock.patch(
                "utils.physloc_reporting._plot_category_breakdown",
                side_effect=RuntimeError("render failed")):
            warnings = write_analysis_bundle(
                root, "model", rows, categories, severity)
            self.assertEqual(len(warnings), 1)
            self.assertTrue(__import__("os").path.exists(
                root + "/analysis/data/metrics_model.csv"))


if __name__ == "__main__":
    unittest.main()
