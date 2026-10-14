
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BINARY_FEATURES,
    CLASS_NAMES,
    HOLDOUT_SIZE,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.exception import AGDPMException
from src.logger import logger


# Columns that must never be used as model inputs.
# They can reveal the target classification.
EXCLUDED_COLUMNS = {
    "Disease",
    "Disease_Subclass",
    TARGET_COLUMN,
}


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw patient records into clean model inputs.
    Used for training AND prediction, so both always match.

    - keeps only approved input features
    - excludes target and leakage-prone columns
    - adds missing input columns as empty columns
    - converts binary values or names (e.g. Female, Yes) to numbers
    - converts features to numeric values; invalid values become NaN
    - clips negative numeric values to zero
    """

    # Ensure INPUT_FEATURES does not contain excluded columns.
    safe_features = [
        column
        for column in INPUT_FEATURES
        if column not in EXCLUDED_COLUMNS
    ]

    # Keep only approved features and add any missing feature columns.
    df = df.reindex(columns=safe_features).copy()

    # Convert binary names to numeric values.
    for column, choices in BINARY_FEATURES.items():
        if column not in df.columns:
            continue

        names = {
            name.lower(): index
            for index, name in enumerate(choices)
        }

        df[column] = df[column].map(
            lambda value: (
                names.get(value.strip().lower(), value)
                if isinstance(value, str)
                else value
            )
        )

    # Convert every input feature to numeric.
    for column in safe_features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Clip negative values for numeric features that are present.
    numeric_columns = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns
    ]

    if numeric_columns:
        df[numeric_columns] = df[numeric_columns].clip(lower=0)

    return df


@dataclass
class DataPreprocessingArtifact:
    X_train: pd.DataFrame
    X_holdout: pd.DataFrame
    y_train: pd.Series
    y_holdout: pd.Series


class DataPreprocessing:

    def initiate_data_preprocessing(
        self,
        df: pd.DataFrame,
    ) -> DataPreprocessingArtifact:

        logger.info("========== DATA PREPROCESSING STARTED ==========")

        try:
            PROCESSED_DATA_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            df = df.copy()

            # Remove completely empty rows.
            df = df.dropna(how="all")

            # Ensure the target column exists.
            if TARGET_COLUMN not in df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' "
                    "was not found in the dataset."
                )

            # Remove records without target labels.
            df = df.dropna(subset=[TARGET_COLUMN])

            # Remove exact duplicate records.
            before = len(df)
            df = df.drop_duplicates()

            logger.info(
                f"Removed {before - len(df)} duplicate rows. "
                f"Rows left: {len(df)}"
            )

            # Validate the text classification labels.
            valid_target_labels = {
                "Single gene inherited disease",
                "Multifactorial genetic disorder",
            }

            y = df[TARGET_COLUMN].astype("string").str.strip()

            unexpected = set(y.dropna().unique()) - valid_target_labels

            if unexpected:
                raise ValueError(
                    f"Unexpected target labels: {sorted(unexpected)}. "
                    f"Expected: {sorted(valid_target_labels)}"
                )

            if y.isna().any():
                raise ValueError(
                    "Target column contains missing labels."
                )

            # Validate that both expected classes exist.
            observed_labels = set(y.unique())

            if observed_labels != valid_target_labels:
                raise ValueError(
                    f"Expected both target classes "
                    f"{sorted(valid_target_labels)}, "
                    f"but found {sorted(observed_labels)}."
                )

            # Explicitly exclude target/leakage columns from model inputs.
            X = clean_features(df)

            # Final safeguard: reject any forbidden input columns.
            forbidden_in_X = EXCLUDED_COLUMNS.intersection(X.columns)

            if forbidden_in_X:
                raise ValueError(
                    "Forbidden columns found in model inputs: "
                    f"{sorted(forbidden_in_X)}"
                )

            logger.info(
                f"Class distribution:\n{y.value_counts().to_string()}"
            )

            logger.info(
                f"Input features ({len(X.columns)}): "
                f"{X.columns.tolist()}"
            )

            # Keep an untouched holdout set for final evaluation.
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X,
                y,
                test_size=HOLDOUT_SIZE,
                stratify=y,
                random_state=RANDOM_STATE,
            )

            # Save the processed training and holdout datasets.
            X_train.to_csv(
                PROCESSED_DATA_DIR / "X_train_clean.csv",
                index=False,
            )

            y_train.to_csv(
                PROCESSED_DATA_DIR / "y_train.csv",
                index=False,
            )

            X_holdout.to_csv(
                PROCESSED_DATA_DIR / "X_holdout_clean.csv",
                index=False,
            )

            y_holdout.to_csv(
                PROCESSED_DATA_DIR / "y_holdout.csv",
                index=False,
            )

            # Save holdout records with their actual labels for inspection.
            holdout_with_answers = X_holdout.copy()
            holdout_with_answers["Actual Disease"] = (
                y_holdout.to_numpy()
            )

            holdout_with_answers.to_csv(
                PROCESSED_DATA_DIR
                / "holdout_patients_with_answers.csv",
                index=False,
            )

            logger.info(
                f"X_train: {X_train.shape}, "
                f"X_holdout: {X_holdout.shape}"
            )

            logger.info(
                "========== DATA PREPROCESSING COMPLETED =========="
            )

            return DataPreprocessingArtifact(
                X_train=X_train.reset_index(drop=True),
                X_holdout=X_holdout.reset_index(drop=True),
                y_train=y_train.reset_index(drop=True),
                y_holdout=y_holdout.reset_index(drop=True),
            )

        except Exception as e:
            logger.exception(
                "Error occurred during data preprocessing."
            )
            raise AGDPMException(e, None) from e
