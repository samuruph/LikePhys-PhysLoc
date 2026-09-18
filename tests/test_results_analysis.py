"""Tests for dataset-neutral evaluation result aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

from analysis.results import analyze_results, collect_records, summarize


class ResultAnalysisTests(unittest.TestCase):
    """Verify that LikePhys and PhysLoc result layouts share one reader."""

    @staticmethod
    def _write_result(path: Path, ratio: float) -> None:
        """Create the minimum persisted evaluator result document."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "misrank_metrics": {
                "base_ppe": {"misrank_ratio": ratio},
            },
        }), encoding="utf-8")

    def test_aggregates_results_from_both_benchmark_directories(self):
        """A common parent produces records and summaries for both datasets."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_result(
                root / "experiment" / "ball_drop" / "results_model.json", .25)
            self._write_result(
                root / "experiment" / "physloc" / "results_model.json", .75)

            records = collect_records(root)
            summaries = analyze_results(root)

            self.assertEqual(len(records), 2)
            self.assertEqual({row["dataset"] for row in records},
                             {"ball_drop", "physloc"})
            overall = next(row for row in summaries if row["dataset"] == "overall")
            self.assertEqual(overall["misrank_ratio"], .5)
            self.assertTrue((root / "misrank_by_variation.csv").is_file())
            self.assertTrue((root / "misrank_summary.csv").is_file())

    def test_dataset_weighting_averages_each_dataset_first(self):
        """Dataset weighting prevents many variations from dominating a summary."""
        records = [
            {"model": "m", "dataset": "many", "misrank_ratio": 0.0},
            {"model": "m", "dataset": "many", "misrank_ratio": 0.0},
            {"model": "m", "dataset": "one", "misrank_ratio": 1.0},
        ]
        summary = summarize(records, weighting="dataset")
        overall = next(row for row in summary if row["dataset"] == "overall")
        self.assertEqual(overall["misrank_ratio"], .5)


if __name__ == "__main__":
    unittest.main()
