"""
tests/test_feature_health.py — Unit tests for FeatureHealthReporter.

Tests: complete valid data, 100% missing, zero variance, single unique value,
JSON report generation.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from feature_health import FeatureHealthReporter, run_feature_health_check


class TestFeatureHealthReporter(unittest.TestCase):
    """FeatureHealthReporter tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = FeatureHealthReporter(report_dir=self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_records(self, count: int, with_variance: bool = True) -> list[dict]:
        """Generate test records with varying feature values."""
        records = []
        for i in range(count):
            rec = {
                "mint": f"mint_{i}",
                "symbol": f"SYM{i}",
                "feature_a": float(i % 10) if with_variance else 5.0,
                "feature_b": i if with_variance else 10,
                "feature_c": 0.0,  # always zero
                "feature_d": 1,     # single unique value
                "inferred_label": True if i % 2 == 0 else False,
            }
            records.append(rec)
        return records

    def test_empty_records_produces_empty_report(self):
        report_json = self.reporter.generate_report(["feature_a"])
        report = json.loads(report_json)
        self.assertEqual(report["total_rows"], 0)

    def test_generates_report_with_all_metrics(self):
        records = self._make_records(20)
        self.reporter.add_records(records)
        report_json = self.reporter.generate_report(["feature_a", "feature_b"])
        report = json.loads(report_json)

        self.assertEqual(report["total_rows"], 20)
        self.assertEqual(report["features_checked"], 2)

        # Check flags structure
        for flag in report["flags"]:
            self.assertIn("feature_name", flag)
            self.assertIn("missing_pct", flag)
            self.assertIn("unique_count", flag)
            self.assertIn("variance", flag)
            self.assertIn("flagged", flag)
            self.assertIn("flag_reason", flag)

    def test_flags_zero_variance_feature(self):
        records = self._make_records(20, with_variance=False)
        self.reporter.add_records(records)
        report_json = self.reporter.generate_report(["feature_a"])
        report = json.loads(report_json)

        flag = report["flags"][0]
        self.assertTrue(flag["flagged"])
        # Constant value triggers single_unique_value before zero_variance check
        self.assertIn(flag["flag_reason"], ("single_unique_value", "zero_variance"))

    def test_flags_single_unique_value(self):
        records = self._make_records(20)
        self.reporter.add_records(records)
        report_json = self.reporter.generate_report(["feature_d"])
        report = json.loads(report_json)

        flag = report["flags"][0]
        self.assertTrue(flag["flagged"])
        self.assertEqual(flag["flag_reason"], "single_unique_value")

    def test_flags_high_missing_rate(self):
        records = [
            {"feature_x": 0.0} for _ in range(15)
        ] + [
            {"feature_x": 1.0} for _ in range(5)
        ]
        # 15/20 = 75% missing
        self.reporter.add_records(records)
        report_json = self.reporter.generate_report(["feature_x"])
        report = json.loads(report_json)

        flag = report["flags"][0]
        self.assertTrue(flag["flagged"])
        self.assertEqual(flag["flag_reason"], "high_missing_rate")

    def test_does_not_flag_with_few_rows(self):
        records = self._make_records(5)  # less than min_rows_for_flag=10
        self.reporter.add_records(records)
        report_json = self.reporter.generate_report(["feature_c"])
        report = json.loads(report_json)

        flag = report["flags"][0]
        self.assertFalse(flag["flagged"])

    def test_auto_discovers_numeric_features(self):
        records = self._make_records(20)
        self.reporter.add_records(records)
        report_json = self.reporter.generate_report()  # no explicit list
        report = json.loads(report_json)

        self.assertGreater(report["features_checked"], 0)
        # Should find feature_a, feature_b, feature_c, feature_d
        feature_names = [f["feature_name"] for f in report["flags"]]
        self.assertIn("feature_a", feature_names)
        # Should NOT include non-numeric keys like "mint"
        self.assertNotIn("mint", feature_names)

    def test_reset_clears_records(self):
        records = self._make_records(20)
        self.reporter.add_records(records)
        self.reporter.reset()
        report_json = self.reporter.generate_report(["feature_a"])
        report = json.loads(report_json)
        self.assertEqual(report["total_rows"], 0)

    def test_report_json_is_valid(self):
        records = self._make_records(15)
        self.reporter.add_records(records)
        report_json = self.reporter.generate_report(["feature_a", "feature_b"])

        # Should be parseable JSON
        report = json.loads(report_json)
        self.assertIsInstance(report, dict)
        self.assertIn("timestamp", report)
        self.assertIn("thresholds", report)

    def test_feature_categories_in_report(self):
        records = self._make_records(15)
        self.reporter.add_records(records)
        categories = {"feature_a": "TEST_CAT_A", "feature_b": "TEST_CAT_B"}
        report_json = self.reporter.generate_report(
            ["feature_a", "feature_b"], categories,
        )
        report = json.loads(report_json)

        for flag in report["flags"]:
            if flag["feature_name"] == "feature_a":
                self.assertEqual(flag["category"], "TEST_CAT_A")
            elif flag["feature_name"] == "feature_b":
                self.assertEqual(flag["category"], "TEST_CAT_B")


class TestRunFeatureHealthCheck(unittest.TestCase):
    """Module-level convenience function tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_check_returns_json(self):
        records = [
            {"f1": float(i), "f2": i} for i in range(20)
        ]
        result = run_feature_health_check(
            records,
            feature_names=["f1", "f2"],
            report_dir=self.temp_dir,
            store_in_db=False,
        )
        self.assertIsInstance(result, str)
        report = json.loads(result)
        self.assertEqual(report["total_rows"], 20)


if __name__ == "__main__":
    unittest.main()
