from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Scripts"))

from _ood_summary_common import (
    extract_prediction_frames,
    load_hybrid_prediction_subset_metrics,
)


class HybridSubsetLabelingTests(unittest.TestCase):
    def test_load_hybrid_prediction_subset_metrics_from_all_predictions_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            test_sets_dir = model_dir / "split_data" / "test_sets"
            predictions_dir = model_dir / "predictions"
            test_sets_dir.mkdir(parents=True)
            predictions_dir.mkdir(parents=True)

            pd.DataFrame({"ID": [1, 2], "target": [10.0, 20.0]}).to_csv(
                test_sets_dir / "test_extrapolation_high20.csv",
                index=False,
            )
            pd.DataFrame({"ID": [3, 4], "target": [30.0, 40.0]}).to_csv(
                test_sets_dir / "test_inner_ood.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "ID": [5, 1, 2, 3, 4],
                    "Dataset": ["Train", "OODTest", "OODTest", "OODTest", "OODTest"],
                    "target_Actual": [0.0, 10.0, 20.0, 30.0, 40.0],
                    "target_Predicted": [0.0, 11.0, 18.0, 33.0, 35.0],
                }
            ).to_csv(predictions_dir / "all_predictions.csv", index=False)

            metrics, audit_rows = load_hybrid_prediction_subset_metrics(
                model_dir,
                "target",
                context={"alloy_family": "Unit", "model": "TabPFN"},
            )

        high_true = [10.0, 20.0]
        high_pred = [11.0, 18.0]
        inner_true = [30.0, 40.0]
        inner_pred = [33.0, 35.0]

        self.assertEqual(metrics["summary_test_extrapolation_high20_n_samples"], 2)
        self.assertTrue(
            math.isclose(
                metrics["summary_test_extrapolation_high20_mae"],
                mean_absolute_error(high_true, high_pred),
            )
        )
        self.assertTrue(
            math.isclose(
                metrics["summary_test_extrapolation_high20_rmse"],
                math.sqrt(mean_squared_error(high_true, high_pred)),
            )
        )
        self.assertTrue(
            math.isclose(
                metrics["summary_test_extrapolation_high20_r2"],
                r2_score(high_true, high_pred),
            )
        )

        self.assertEqual(metrics["summary_test_inner_ood_n_samples"], 2)
        self.assertTrue(math.isclose(metrics["summary_test_inner_ood_mae"], mean_absolute_error(inner_true, inner_pred)))
        self.assertTrue(
            math.isclose(
                metrics["summary_test_inner_ood_rmse"],
                math.sqrt(mean_squared_error(inner_true, inner_pred)),
            )
        )
        self.assertTrue(math.isclose(metrics["summary_test_inner_ood_r2"], r2_score(inner_true, inner_pred)))

        self.assertEqual(len(audit_rows), 2)
        self.assertEqual({row["status"] for row in audit_rows}, {"complete"})
        self.assertEqual({row["source_mode"] for row in audit_rows}, {"all_predictions_id_match"})
        self.assertTrue(all(row["expected_id_count"] == row["matched_prediction_count"] for row in audit_rows))
        self.assertTrue(all(row["missing_id_count"] == 0 for row in audit_rows))

    def test_extract_prediction_frames_recognizes_hybrid_subset_dataset_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_file = Path(tmpdir) / "subset_predictions.csv"
            pd.DataFrame(
                {
                    "ID": [1, 2],
                    "Dataset": ["Train", "test_extrapolation_high20"],
                    "target_Actual": [1.0, 2.0],
                    "target_Predicted": [1.1, 1.9],
                }
            ).to_csv(prediction_file, index=False)

            _, test_df, dataset_col, actual_col, pred_col = extract_prediction_frames(prediction_file, "target")

        self.assertEqual(dataset_col, "Dataset")
        self.assertEqual(actual_col, "target_Actual")
        self.assertEqual(pred_col, "target_Predicted")
        self.assertIsNotNone(test_df)
        assert test_df is not None
        self.assertEqual(test_df["ID"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
