
import unittest
import numpy as np
import pandas as pd
from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # --- Test 1: testing the select_features functions, checking to see that it picks only the requested numeric columns ---
    # If non-numeric or unintended columns are selected, clustering distances
    # can be distorted, leading to incorrect cluster assignments.
    def test_select_features_correct_columns(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ['X', 'Y', 'Z'],
            "C": [7, 8, 9]
        })
        print("\n[TEST 1] Attempting to select columns including a non-numeric one...")
        with self.assertRaises(TypeError) as context:
            select_features(df, ["A", "B", "C"])
        print("[TEST 1] Caught expected TypeError:", context.exception)

    # --- Test 2: standardise_features produces zero mean and unit variance ---
    # If features are not properly standardised, distance-based algorithms
    # like k-means will give biased results because features on larger scales dominate.
    def test_standardise_features_zero_mean_unit_variance(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        X_std = standardise_features(X)
        mean = X_std.mean(axis=0)
        std = X_std.std(axis=0, ddof=0)
        print("\n[TEST 2] Standardised data:\n", X_std)
        print("[TEST 2] Column means:", mean)
        print("[TEST 2] Column std devs:", std)
        self.assertTrue(np.allclose(mean, 0.0, atol=1e-8))
        self.assertTrue(np.allclose(std, 1.0, atol=1e-8))
        print("[TEST 2] Passed: zero mean and unit variance confirmed")

    # --- Test 3: checking standardise_features when it handles columns with zero variance (i.e., whole columns are constant) ---
    # Columns with zero variance are common in real datasets (e.g., binary flags),
    # and must not crash the scaler. They should return zeros after standardisation.
    def test_standardise_features_constant_column(self):
        X = np.array([[5, 1], [5, 2], [5, 3]], dtype=float)  # first column constant
        X_std = standardise_features(X)
        print("\n[TEST 3] Standardised data with constant column:\n", X_std)
        mean = X_std[:, 1].mean()
        std = X_std[:, 1].std(ddof=0)
        print("[TEST 3] Non-constant column mean:", mean)
        print("[TEST 3] Non-constant column std dev:", std)
        self.assertTrue(np.allclose(X_std[:, 0], 0.0))
        self.assertTrue(np.allclose(mean, 0.0, atol=1e-8))
        self.assertTrue(np.allclose(std, 1.0, atol=1e-8))
        print("[TEST 3] Passed: constant column handled correctly")


if __name__ == "__main__":
    unittest.main()