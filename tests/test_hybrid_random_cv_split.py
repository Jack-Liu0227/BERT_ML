from __future__ import annotations

import unittest

import pandas as pd

from src.data_processing.strength_ood_common import (
    HYBRID_INNER_TEST_SET_NAME,
    HYBRID_OUTER_TEST_SET_NAME,
    prepare_hybrid_extrapolation_split,
)


class HybridRandomCVSplitTests(unittest.TestCase):
    def test_hybrid_random_cv_uses_fixed_high20_outer_and_random_inner_folds(self) -> None:
        df = pd.DataFrame(
            {
                "ID": list(range(20)),
                "Al(wt%)": [float(index) for index in range(20)],
                "temperature": [float(400 + index) for index in range(20)],
                "target": [float(index) for index in range(20)],
            }
        )

        folds = prepare_hybrid_extrapolation_split(
            df=df,
            target_col="target",
            split_strategy="hybrid_extrapolation_random_cv",
            inner_strategy="random_cv",
            outer_test_ratio=0.2,
            inner_test_ratio=0.2,
            random_state=42,
            cluster_count=5,
        )

        self.assertIsInstance(folds, list)
        self.assertEqual(len(folds), 5)

        expected_outer_ids = {16, 17, 18, 19}
        all_inner_ids: set[int] = set()
        for fold_index, fold in enumerate(folds):
            split = fold.split
            self.assertEqual(split.split_strategy, "hybrid_extrapolation_random_cv")
            self.assertEqual(split.summary["inner_strategy"], "random_cv")
            self.assertEqual(split.summary["outer_test_size"], 4)
            self.assertEqual(split.summary["fold_index"], fold_index)
            self.assertEqual(split.summary["outer_fold_count"], 5)

            outer_ids = set(split.test_sets[HYBRID_OUTER_TEST_SET_NAME]["ID"].astype(int).tolist())
            inner_ids = set(split.test_sets[HYBRID_INNER_TEST_SET_NAME]["ID"].astype(int).tolist())
            combined_ids = set(split.test_df["ID"].astype(int).tolist())
            train_ids = set(split.train_df["ID"].astype(int).tolist())

            self.assertEqual(split.summary["inner_test_size"], len(inner_ids))
            self.assertEqual(split.summary["combined_test_size"], len(combined_ids))
            self.assertEqual(outer_ids, expected_outer_ids)
            self.assertTrue(inner_ids.isdisjoint(expected_outer_ids))
            self.assertEqual(combined_ids, outer_ids | inner_ids)
            self.assertTrue(train_ids.isdisjoint(combined_ids))
            all_inner_ids.update(inner_ids)

        self.assertEqual(all_inner_ids, set(range(16)))


if __name__ == "__main__":
    unittest.main()
