from dataclasses import dataclass

import pandas as pd

from src.config import (
    BINARY_FEATURES,
    CLASS_NAMES,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    VALIDATION_DIR,
)
from src.exception import AGDPMException
from src.logger import logger


@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: str


class DataValidation:
    """Checks the raw data and writes a readable quality report."""

    def initiate_data_validation(self, df: pd.DataFrame) -> DataValidationArtifact:

        logger.info("========== DATA VALIDATION STARTED ==========")

        try:
            VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
            report_path = VALIDATION_DIR / "validation_report.txt"
            issues = []

            missing_columns = [
                c for c in INPUT_FEATURES + [TARGET_COLUMN]
                if c not in df.columns
            ]

            if missing_columns:
                issues.append(f"Missing columns: {missing_columns}")

            empty_rows = int(df.isna().all(axis=1).sum())
            duplicate_rows = int(df.duplicated().sum())
            missing_values = df.isna().sum()

            if empty_rows:
                issues.append(f"{empty_rows} completely empty rows.")

            if duplicate_rows:
                issues.append(f"{duplicate_rows} duplicate rows.")

            if missing_values.sum():
                issues.append(f"{int(missing_values.sum())} missing values.")

            # Validate text target labels
            if TARGET_COLUMN in df.columns:
                labels = set(df[TARGET_COLUMN].dropna().unique())

                expected_labels = {
                    "Single gene inherited disease",
                    "Multifactorial genetic disorder",
                }

                unexpected = labels - expected_labels

                if unexpected:
                    issues.append(
                        f"Unexpected target labels: {sorted(unexpected)}"
                    )

                if df[TARGET_COLUMN].isna().any():
                    issues.append(
                        f"Target column '{TARGET_COLUMN}' contains missing labels."
                    )

            # Validate binary features
            for column, choices in BINARY_FEATURES.items():
                if column in df.columns:
                    bad = ~df[column].dropna().isin(range(len(choices)))

                    if bad.any():
                        issues.append(
                            f"'{column}' has {int(bad.sum())} invalid binary values."
                        )

            numeric_columns = [
                c for c in NUMERIC_FEATURES if c in df.columns
            ]
            numeric_summary = df[numeric_columns].describe().T

            with open(report_path, "w", encoding="utf-8") as file:
                file.write("DATA VALIDATION REPORT\n" + "=" * 60 + "\n\n")
                file.write(f"Shape: {df.shape}\n")
                file.write(f"Empty rows: {empty_rows}\n")
                file.write(f"Duplicate rows: {duplicate_rows}\n\n")

                file.write("Missing values by column\n" + "-" * 60 + "\n")
                file.write(missing_values.to_string() + "\n\n")

                if TARGET_COLUMN in df.columns:
                    counts = df[TARGET_COLUMN].value_counts()

                    file.write("Class distribution\n" + "-" * 60 + "\n")

                    for label, count in counts.items():
                        file.write(f"{label:<45} {count}\n")

                    file.write("\n")

                file.write("Numeric features\n" + "-" * 60 + "\n")
                file.write(numeric_summary.to_string() + "\n\n")

                file.write("Issues\n" + "-" * 60 + "\n")
                file.write(
                    "\n".join(f"- {issue}" for issue in issues)
                    if issues
                    else "No issues found."
                )
                file.write("\n")

            for issue in issues:
                logger.warning(f"Validation issue: {issue}")

            logger.info(f"Validation report saved: {report_path}")
            logger.info("========== DATA VALIDATION COMPLETED ==========")

            return DataValidationArtifact(
                validation_status=not issues,
                validation_report_path=str(report_path),
            )

        except Exception as e:
            logger.exception("Error occurred during data validation.")
            raise AGDPMException(e, None) from e