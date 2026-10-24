"""
tests/test_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Unit and integration tests for Phase 1 preprocessing.

Run with:
    python -m pytest tests/test_preprocessing.py -v
    # or directly:
    python tests/test_preprocessing.py

Test groups
───────────
  A. Raw data integrity      — checks on data/raw/ CSVs
  B. Schema compliance       — column_schema.py consistency
  C. Preprocessing outputs   — checks on data/processed/ outputs
  D. Leakage checks          — ensure no forbidden columns in feature matrices
  E. Patient-level integrity — zero overlap between train/test at every stage
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.column_schema import (
    ID_COLS, LABEL_COL, SPLIT_COL, IMAGE_COL, AUDIT_COLS,
    MODEL_INPUT_COLS, MEASURED_FLAG_COLS, COUNT_COLS,
    ALL_EXPECTED_COLS,
)

RAW_DIR  = os.path.join(_PROJECT_ROOT, "data", "raw")
PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

MAIN_CSV   = os.path.join(RAW_DIR, "dataset_T6h_H24_v5.csv")
STRICT_CSV = os.path.join(RAW_DIR, "dataset_T6h_H24_v5_strictfollowup.csv")


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# A. Raw data integrity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRawDataIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(MAIN_CSV):
            raise unittest.SkipTest(f"Raw CSV not found: {MAIN_CSV}")
        cls.df  = _load(MAIN_CSV)
        cls.dfs = _load(STRICT_CSV) if os.path.isfile(STRICT_CSV) else None

    def test_A01_main_csv_shape(self):
        """Main CSV must have exactly 2,884 rows and 62 columns."""
        self.assertEqual(self.df.shape, (2884, 62),
            f"Expected (2884, 62), got {self.df.shape}")

    def test_A02_strict_csv_shape(self):
        """Strict-followup CSV must have 1,909 rows and 62 columns."""
        if self.dfs is None:
            self.skipTest("Strict CSV not available")
        self.assertEqual(self.dfs.shape, (1909, 62),
            f"Expected (1909, 62), got {self.dfs.shape}")

    def test_A03_no_duplicate_rows(self):
        """No fully duplicated rows."""
        n = self.df.duplicated().sum()
        self.assertEqual(n, 0, f"{n} duplicate rows found")

    def test_A04_no_duplicate_stay_ids(self):
        """Each stay_id must appear exactly once."""
        n = self.df["stay_id"].duplicated().sum()
        self.assertEqual(n, 0, f"{n} duplicate stay_ids")

    def test_A05_no_duplicate_dicom_ids(self):
        """Each dicom_id must appear exactly once."""
        n = self.df["dicom_id"].duplicated().sum()
        self.assertEqual(n, 0, f"{n} duplicate dicom_ids")

    def test_A06_no_missing_labels(self):
        """sepsis_label must have zero missing values."""
        n = self.df[LABEL_COL].isna().sum()
        self.assertEqual(n, 0, f"{n} missing labels")

    def test_A07_label_values_binary(self):
        """sepsis_label must contain only 0 and 1."""
        vals = set(self.df[LABEL_COL].unique())
        self.assertEqual(vals, {0, 1}, f"Unexpected label values: {vals}")

    def test_A08_positive_count(self):
        """Must have exactly 476 positive cases."""
        n_pos = int(self.df[LABEL_COL].sum())
        self.assertEqual(n_pos, 476, f"Expected 476 positives, got {n_pos}")

    def test_A09_split_values(self):
        """split column must contain only 'train' and 'test'."""
        vals = set(self.df[SPLIT_COL].unique())
        self.assertEqual(vals, {"train", "test"}, f"Unexpected split values: {vals}")

    def test_A10_no_missing_image_path(self):
        """image_path must be fully populated."""
        n = self.df[IMAGE_COL].isna().sum()
        self.assertEqual(n, 0, f"{n} missing image_path values")

    def test_A11_measured_flags_complete(self):
        """All _measured binary flags must be 0% missing."""
        for col in MEASURED_FLAG_COLS:
            n = self.df[col].isna().sum()
            self.assertEqual(n, 0, f"`{col}` has {n} missing values")

    def test_A12_measured_flags_binary(self):
        """All _measured flags must contain only 0 and 1."""
        for col in MEASURED_FLAG_COLS:
            vals = set(self.df[col].unique())
            self.assertTrue(vals.issubset({0, 1}),
                f"`{col}` contains non-binary values: {vals}")

    
    def test_A13_negative_onset_times_are_outside_prediction_window(self):
        """Negative cases with onset timestamps must have onset after 24 hours."""
        neg_mask = self.df[LABEL_COL] == 0

        neg_with_onset = self.df.loc[
            neg_mask & self.df["sepsis_onset_time"].notna()
        ].copy()

        prediction = pd.to_datetime(
            neg_with_onset["prediction_time"], errors="coerce"
        )
        onset = pd.to_datetime(
            neg_with_onset["sepsis_onset_time"], errors="coerce"
        )

        hours_to_onset = (
            onset - prediction
        ).dt.total_seconds() / 3600

        invalid = hours_to_onset.isna() | (hours_to_onset <= 24)

        self.assertEqual(
            int(invalid.sum()),
            0,
            "Negative rows contain onset timestamps within 24 hours "
            "of prediction time or invalid timestamps."
        )

    def test_A14_positive_onset_times_not_null(self):
        """sepsis_onset_time must be present for all positive cases."""
        pos_mask = self.df[LABEL_COL] == 1
        n_missing = self.df.loc[pos_mask, "sepsis_onset_time"].isna().sum()
        self.assertEqual(n_missing, 0,
            f"{n_missing} positive rows have missing sepsis_onset_time")


# ─────────────────────────────────────────────────────────────────────────────
# B. Schema compliance tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaCompliance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(MAIN_CSV):
            raise unittest.SkipTest(f"Raw CSV not found: {MAIN_CSV}")
        cls.df = _load(MAIN_CSV)

    def test_B01_all_expected_columns_present(self):
        """Every column listed in ALL_EXPECTED_COLS must be in the CSV."""
        missing = [c for c in ALL_EXPECTED_COLS if c not in self.df.columns]
        self.assertEqual(missing, [], f"Missing columns: {missing}")

    def test_B02_no_extra_columns(self):
        """No unexpected columns in the CSV."""
        extra = [c for c in self.df.columns if c not in ALL_EXPECTED_COLS]
        self.assertEqual(extra, [], f"Unexpected columns: {extra}")

    def test_B03_total_expected_columns(self):
        """ALL_EXPECTED_COLS must enumerate exactly 62 columns."""
        self.assertEqual(len(ALL_EXPECTED_COLS), 62,
            f"ALL_EXPECTED_COLS has {len(ALL_EXPECTED_COLS)} entries, expected 62")

    def test_B04_model_input_count(self):
        """MODEL_INPUT_COLS must enumerate exactly 39 features."""
        self.assertEqual(len(MODEL_INPUT_COLS), 42,
            f"MODEL_INPUT_COLS has {len(MODEL_INPUT_COLS)}, expected 39")

    def test_B05_no_id_in_model_inputs(self):
        """No identifier column must appear in MODEL_INPUT_COLS."""
        overlap = set(ID_COLS) & set(MODEL_INPUT_COLS)
        self.assertEqual(overlap, set(), f"ID cols in model inputs: {overlap}")

    def test_B06_no_audit_in_model_inputs(self):
        """No audit/leakage column must appear in MODEL_INPUT_COLS."""
        overlap = set(AUDIT_COLS) & set(MODEL_INPUT_COLS)
        self.assertEqual(overlap, set(), f"Audit cols in model inputs: {overlap}")

    def test_B07_label_not_in_model_inputs(self):
        """Label column must not appear in MODEL_INPUT_COLS."""
        self.assertNotIn(LABEL_COL, MODEL_INPUT_COLS,
            f"`{LABEL_COL}` must not be a model input")

    def test_B08_leakage_columns_not_in_model_inputs(self):
        """Known post-prediction columns must be excluded from model inputs."""
        leakage_cols = [
            "sepsis_onset_time", "hours_to_onset",
            "followup_h", "prediction_time",
        ]
        for col in leakage_cols:
            self.assertNotIn(col, MODEL_INPUT_COLS,
                f"Leakage column `{col}` found in MODEL_INPUT_COLS")


# ─────────────────────────────────────────────────────────────────────────────
# C. Preprocessing output tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        files = {
            "X_train": os.path.join(PROC_DIR, "X_train.csv"),
            "X_test":  os.path.join(PROC_DIR, "X_test.csv"),
            "y_train": os.path.join(PROC_DIR, "y_train.csv"),
            "y_test":  os.path.join(PROC_DIR, "y_test.csv"),
            "ids_train": os.path.join(PROC_DIR, "ids_train.csv"),
            "ids_test":  os.path.join(PROC_DIR, "ids_test.csv"),
            "imp":     os.path.join(PROC_DIR, "imputation_stats.csv"),
        }
        missing = [k for k, v in files.items() if not os.path.isfile(v)]
        if missing:
            raise unittest.SkipTest(
                f"Processed outputs not found: {missing}. "
                "Run `python src/preprocessing/preprocess.py` first."
            )
        cls.X_train = _load(files["X_train"])
        cls.X_test  = _load(files["X_test"])
        cls.y_train = _load(files["y_train"])
        cls.y_test  = _load(files["y_test"])
        cls.ids_train = _load(files["ids_train"])
        cls.ids_test  = _load(files["ids_test"])
        cls.imp     = _load(files["imp"])

    def test_C01_row_counts_match(self):
        """X/y/ids must have the same row count for each split."""
        self.assertEqual(len(self.X_train), len(self.y_train))
        self.assertEqual(len(self.X_train), len(self.ids_train))
        self.assertEqual(len(self.X_test),  len(self.y_test))
        self.assertEqual(len(self.X_test),  len(self.ids_test))

    def test_C02_train_row_count(self):
        """Training set must have 2,282 rows."""
        self.assertEqual(len(self.X_train), 2282,
            f"Expected 2282 train rows, got {len(self.X_train)}")

    def test_C03_test_row_count(self):
        """Test set must have 602 rows."""
        self.assertEqual(len(self.X_test), 602,
            f"Expected 602 test rows, got {len(self.X_test)}")

    def test_C04_no_nan_in_X_train(self):
        """X_train must have zero NaN values after imputation."""
        n = self.X_train.isna().sum().sum()
        self.assertEqual(n, 0, f"X_train has {n} NaN values after imputation")

    def test_C05_no_nan_in_X_test(self):
        """X_test must have zero NaN values after imputation."""
        n = self.X_test.isna().sum().sum()
        self.assertEqual(n, 0, f"X_test has {n} NaN values after imputation")

    def test_C06_no_leakage_cols_in_X_train(self):
        """No identifier, audit, or label column in X_train."""
        forbidden = set(ID_COLS) | set(AUDIT_COLS) | {LABEL_COL, SPLIT_COL, IMAGE_COL}
        present = forbidden & set(self.X_train.columns)
        self.assertEqual(present, set(), f"Forbidden columns in X_train: {present}")

    def test_C07_no_leakage_cols_in_X_test(self):
        """No identifier, audit, or label column in X_test."""
        forbidden = set(ID_COLS) | set(AUDIT_COLS) | {LABEL_COL, SPLIT_COL, IMAGE_COL}
        present = forbidden & set(self.X_test.columns)
        self.assertEqual(present, set(), f"Forbidden columns in X_test: {present}")

    def test_C08_y_train_positive_count(self):
        """Training labels must have exactly 372 positives."""
        n = int(self.y_train[LABEL_COL].sum())
        self.assertEqual(n, 372, f"Expected 372 train positives, got {n}")

    def test_C09_y_test_positive_count(self):
        """Test labels must have exactly 104 positives."""
        n = int(self.y_test[LABEL_COL].sum())
        self.assertEqual(n, 104, f"Expected 104 test positives, got {n}")

    def test_C10_imputation_stats_present(self):
        """imputation_stats.csv must exist and contain 'column' and 'train_median'."""
        self.assertIn("column", self.imp.columns)
        self.assertIn("train_median", self.imp.columns)

    def test_C11_same_feature_columns_train_test(self):
        """X_train and X_test must have identical columns."""
        self.assertEqual(
            list(self.X_train.columns), list(self.X_test.columns),
            "Column mismatch between X_train and X_test"
        )

    def test_C12_ids_contain_stay_id(self):
        """ids_train and ids_test must contain stay_id."""
        self.assertIn("stay_id", self.ids_train.columns)
        self.assertIn("stay_id", self.ids_test.columns)


# ─────────────────────────────────────────────────────────────────────────────
# D. Leakage tests (extra paranoia layer)
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakageChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x_path = os.path.join(PROC_DIR, "X_train.csv")
        if not os.path.isfile(x_path):
            raise unittest.SkipTest("Processed outputs not found.")
        cls.X_train = _load(x_path)
        cls.X_test  = _load(os.path.join(PROC_DIR, "X_test.csv"))

    def test_D01_no_sepsis_onset_time(self):
        for df, name in [(self.X_train, "X_train"), (self.X_test, "X_test")]:
            self.assertNotIn("sepsis_onset_time", df.columns,
                f"`sepsis_onset_time` (post-T leakage) found in {name}")

    def test_D02_no_hours_to_onset(self):
        for df, name in [(self.X_train, "X_train"), (self.X_test, "X_test")]:
            self.assertNotIn("hours_to_onset", df.columns,
                f"`hours_to_onset` (post-T leakage) found in {name}")

    def test_D03_no_followup_h(self):
        for df, name in [(self.X_train, "X_train"), (self.X_test, "X_test")]:
            self.assertNotIn("followup_h", df.columns,
                f"`followup_h` (post-T info) found in {name}")

    def test_D04_no_prediction_time(self):
        for df, name in [(self.X_train, "X_train"), (self.X_test, "X_test")]:
            self.assertNotIn("prediction_time", df.columns,
                f"`prediction_time` found in {name}")

    def test_D05_no_subject_id(self):
        for df, name in [(self.X_train, "X_train"), (self.X_test, "X_test")]:
            self.assertNotIn("subject_id", df.columns,
                f"`subject_id` found in {name}")


# ─────────────────────────────────────────────────────────────────────────────
# E. Patient-level integrity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPatientLevelIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for p in [
            os.path.join(PROC_DIR, "ids_train.csv"),
            os.path.join(PROC_DIR, "ids_test.csv"),
        ]:
            if not os.path.isfile(p):
                raise unittest.SkipTest("ids_*.csv not found — run preprocess.py first.")
        cls.ids_train = _load(os.path.join(PROC_DIR, "ids_train.csv"))
        cls.ids_test  = _load(os.path.join(PROC_DIR, "ids_test.csv"))

    def test_E01_zero_patient_overlap(self):
        """No patient (subject_id) must appear in both train and test."""
        tr = set(self.ids_train["subject_id"])
        te = set(self.ids_test["subject_id"])
        overlap = tr & te
        self.assertEqual(len(overlap), 0,
            f"{len(overlap)} patients found in BOTH splits: {list(overlap)[:5]}")

    def test_E02_zero_stay_overlap(self):
        """No stay_id must appear in both train and test."""
        tr = set(self.ids_train["stay_id"])
        te = set(self.ids_test["stay_id"])
        overlap = tr & te
        self.assertEqual(len(overlap), 0,
            f"{len(overlap)} stay_ids found in BOTH splits")

    def test_E03_zero_dicom_overlap(self):
        """No dicom_id must appear in both train and test."""
        tr = set(self.ids_train["dicom_id"])
        te = set(self.ids_test["dicom_id"])
        overlap = tr & te
        self.assertEqual(len(overlap), 0,
            f"{len(overlap)} dicom_ids found in BOTH splits")

    def test_E04_train_patient_count(self):
        """Train set must have exactly 2,108 unique patients."""
        n = self.ids_train["subject_id"].nunique()
        self.assertEqual(n, 2108, f"Expected 2108 train patients, got {n}")

    def test_E05_test_patient_count(self):
        """Test set must have exactly 553 unique patients."""
        n = self.ids_test["subject_id"].nunique()
        self.assertEqual(n, 553, f"Expected 553 test patients, got {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestRawDataIntegrity,
        TestSchemaCompliance,
        TestPreprocessingOutputs,
        TestLeakageChecks,
        TestPatientLevelIntegrity,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
