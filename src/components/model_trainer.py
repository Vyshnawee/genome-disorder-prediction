import json
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.components.feature_engineering import build_preprocessor
from src.config import (
    BINARY_FEATURES,
    CLASS_INFO,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    CV_FOLDS,
    FEATURE_LABELS,
    FORM_SECTIONS,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_INFO_FILE,
    NORMAL_RANGES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.exception import AGDPMException
from src.logger import logger


@dataclass
class ModelTrainerArtifact:
    model: Pipeline
    model_name: str
    cv_results: pd.DataFrame
    model_path: str


def encode_target(y: pd.Series) -> np.ndarray:
    """Convert target labels into numeric class IDs."""
    class_to_index = {
        str(name).strip(): index
        for index, name in enumerate(CLASS_NAMES)
    }

    y_clean = y.astype("string").str.strip()
    encoded = y_clean.map(class_to_index)

    if encoded.isna().any():
        unexpected = sorted(
            y_clean[encoded.isna()].dropna().unique().tolist()
        )
        raise ValueError(
            f"Unknown target labels: {unexpected}. "
            f"Expected labels: {list(CLASS_NAMES)}"
        )

    return encoded.astype(int).to_numpy()


class ModelTrainer:
    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Strongly regularized model configurations. These settings do not
        # guarantee a specific accuracy range.
        self.models = {
            "Logistic Regression": LogisticRegression(
                C=0.00001,
                max_iter=500,
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=10,
                max_depth=1,
                min_samples_split=60,
                min_samples_leaf=50,
                max_features=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
                class_weight="balanced",
            ),
            "XGBoost": XGBClassifier(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=10,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=5.0,
                reg_lambda=10.0,
                eval_metric="logloss",
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            "CatBoost": CatBoostClassifier(
                iterations=5,
                depth=1,
                learning_rate=0.001,
                l2_leaf_reg=100,
                loss_function="Logloss",
                random_seed=RANDOM_STATE,
                thread_count=1,
                verbose=0,
                allow_writing_files=False,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=5,
                max_depth=1,
                num_leaves=2,
                learning_rate=0.001,
                min_child_samples=100,
                reg_alpha=50.0,
                reg_lambda=100.0,
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
                class_weight="balanced",
            ),
            "AdaBoost": AdaBoostClassifier(
                random_state=RANDOM_STATE,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=10,
                max_depth=1,
                min_samples_split=60,
                min_samples_leaf=50,
                max_features=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
                class_weight="balanced",
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=1,
                min_samples_split=200,
                min_samples_leaf=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(2,),
                activation="logistic",
                solver="adam",
                alpha=100.0,
                max_iter=50,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        }

        self.cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    @staticmethod
    def build_pipeline(model) -> Pipeline:
        """Fit preprocessing separately within each CV fold."""
        return Pipeline(
            steps=[
                ("preprocessing", build_preprocessor()),
                ("model", model),
            ]
        )

    def build_model_info(
        self,
        X_train: pd.DataFrame,
        model_name: str,
    ) -> dict:
        """Build metadata for the prediction form, using available features only."""
        fields = {}

        for column in NUMERIC_FEATURES:
            if column not in X_train.columns:
                continue

            values = pd.to_numeric(
                X_train[column], errors="coerce"
            ).dropna()

            if values.empty:
                logger.warning(
                    f"Skipping form metadata for empty feature: {column}"
                )
                continue

            is_integer = bool((values % 1 == 0).all())
            fields[column] = {
                "type": "number",
                "label": FEATURE_LABELS.get(column, column),
                "min": round(float(values.min()), 2),
                "max": round(float(values.max()), 2),
                "step": 1 if is_integer else 0.01,
                "hint": NORMAL_RANGES.get(column, ""),
            }

        for column, choices in BINARY_FEATURES.items():
            if column not in X_train.columns:
                continue

            fields[column] = {
                "type": "binary",
                "label": FEATURE_LABELS.get(column, column),
                "options": [
                    {"value": index, "label": name}
                    for index, name in enumerate(choices)
                ],
            }

        # Keep only section fields that have metadata. This prevents the
        # frontend from reading .type on an undefined field entry.
        sections = []
        for section in FORM_SECTIONS:
            valid_fields = [
                field_name
                for field_name in section.get("fields", [])
                if field_name in fields
            ]
            if valid_fields:
                sections.append({**section, "fields": valid_fields})

        return {
            "model_name": model_name,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "classes": list(CLASS_NAMES),
            "class_info": CLASS_INFO,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "sections": sections,
            "fields": fields,
        }

    def initiate_model_training(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> ModelTrainerArtifact:
        logger.info("========== MODEL TRAINING STARTED ==========")

        try:
            X_train = X_train.reset_index(drop=True).copy()
            y_train = pd.Series(y_train).reset_index(drop=True)

            if len(X_train) != len(y_train):
                raise ValueError(
                    "X_train and y_train have different numbers of rows."
                )
            if X_train.empty:
                raise ValueError("X_train is empty.")
            if X_train.columns.duplicated().any():
                duplicates = X_train.columns[
                    X_train.columns.duplicated()
                ].tolist()
                raise ValueError(
                    f"Duplicate feature columns found: {duplicates}"
                )
            if y_train.isna().any():
                raise ValueError("The target column contains missing values.")

            y_encoded = encode_target(y_train)
            observed_classes = set(np.unique(y_encoded))
            expected_classes = set(range(len(CLASS_NAMES)))

            if observed_classes != expected_classes:
                raise ValueError(
                    f"Training classes are {sorted(observed_classes)}, "
                    f"but CLASS_NAMES defines {sorted(expected_classes)}. "
                    "Check CLASS_NAMES and the target column."
                )

            class_counts = pd.Series(y_encoded).value_counts()
            if class_counts.min() < self.cv.n_splits:
                raise ValueError(
                    f"Each class must have at least {self.cv.n_splits} "
                    "samples for stratified cross-validation. "
                    f"Class counts: {class_counts.to_dict()}"
                )

            logger.info(
                "Training class distribution:\n"
                f"{y_train.value_counts().to_string()}"
            )

            # Diagnostic: estimate whether any individual feature is unusually
            # predictive. This is a diagnostic, not a proof of leakage.
            single_feature_scores = {}
            for col in X_train.columns:
                x = (
                    pd.to_numeric(X_train[col], errors="coerce")
                    .fillna(-999)
                    .to_frame()
                )
                single_feature_scores[col] = cross_val_score(
                    DecisionTreeClassifier(
                        max_depth=1,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                    x,
                    y_encoded,
                    cv=self.cv,
                    scoring="accuracy",
                    n_jobs=1,
                ).mean()

            logger.info(
                "Single-feature CV accuracy (top 10):\n"
                + pd.Series(single_feature_scores)
                .sort_values(ascending=False)
                .head(10)
                .to_string()
            )

            results = []
            scoring = {
                "accuracy": "accuracy",
                "macro_f1": "f1_macro",
                "weighted_f1": "f1_weighted",
                "neg_log_loss": "neg_log_loss",
            }

            for model_name, model in self.models.items():
                logger.info(f"Cross-validating: {model_name}")
                pipeline = self.build_pipeline(model)

                cv_results = cross_validate(
                    pipeline,
                    X_train,
                    y_encoded,
                    cv=self.cv,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                    error_score="raise",
                )

                row = {
                    "Model": model_name,
                    "Train Accuracy": float(
                        cv_results["train_accuracy"].mean()
                    ),
                    "CV Accuracy": float(
                        cv_results["test_accuracy"].mean()
                    ),
                    "CV Accuracy Std": float(
                        cv_results["test_accuracy"].std()
                    ),
                    "CV Macro F1": float(
                        cv_results["test_macro_f1"].mean()
                    ),
                    "CV Weighted F1": float(
                        cv_results["test_weighted_f1"].mean()
                    ),
                    "CV Log Loss": float(
                        -cv_results["test_neg_log_loss"].mean()
                    ),
                }
                results.append(row)

                logger.info(
                    f"{model_name} | "
                    f"Train Accuracy={row['Train Accuracy']:.4f} | "
                    f"CV Accuracy={row['CV Accuracy']:.4f} | "
                    f"Macro F1={row['CV Macro F1']:.4f} | "
                    f"Weighted F1={row['CV Weighted F1']:.4f} | "
                    f"Log Loss={row['CV Log Loss']:.4f}"
                )

            results_df = (
                pd.DataFrame(results)
                .sort_values(
                    by=["CV Log Loss", "CV Accuracy"],
                    ascending=[True, False],
                )
                .reset_index(drop=True)
            )

            # Save the standard report when possible. If Windows has locked it
            # (for example, because it is open in Excel), save a timestamped copy.
            comparison_path = REPORTS_DIR / "model_comparison.csv"
            try:
                results_df.to_csv(comparison_path, index=False)
            except PermissionError:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                comparison_path = (
                    REPORTS_DIR / f"model_comparison_{timestamp}.csv"
                )
                logger.warning(
                    "model_comparison.csv is locked. Saving comparison to "
                    f"{comparison_path} instead."
                )
                results_df.to_csv(comparison_path, index=False)

            best_model_name = str(results_df.loc[0, "Model"])
            logger.info(
                f"Selected model by CV log loss: {best_model_name}"
            )
            logger.info(f"Model comparison saved: {comparison_path}")

            # Fit the selected model on the training split only.
            final_pipeline = self.build_pipeline(self.models[best_model_name])
            final_pipeline.fit(X_train, y_encoded)

            joblib.dump(final_pipeline, MODEL_FILE)
            logger.info(f"Model saved: {MODEL_FILE}")

            model_info = self.build_model_info(X_train, best_model_name)
            with open(MODEL_INFO_FILE, "w", encoding="utf-8") as file:
                json.dump(
                    model_info,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            logger.info(f"Model info saved: {MODEL_INFO_FILE}")
            logger.info("========== MODEL TRAINING COMPLETED ==========")

            return ModelTrainerArtifact(
                model=final_pipeline,
                model_name=best_model_name,
                cv_results=results_df,
                model_path=str(MODEL_FILE),
            )

        except Exception as e:
            logger.exception("Error occurred during model training.")
            raise AGDPMException(e, None) from e
