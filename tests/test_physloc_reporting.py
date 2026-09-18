import tempfile
import unittest

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
                root + "/analysis/plots/model/severity_base_ppe.png"))


if __name__ == "__main__":
    unittest.main()
