import json
from dataclasses import dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    FEATURE_LABELS,
    METRICS_FILE,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.exception import AGDPMException
from src.logger import logger


@dataclass
class ModelEvaluationArtifact:
    evaluation_metrics_path: str
    classification_report_path: str
    confusion_matrix_path: str
    confusion_matrix_plot_path: str
    metrics: dict


class ModelEvaluation:
    """Evaluate a fitted model and write both evaluation and dashboard reports."""

    MODEL_DISPLAY_NAMES = {
        "SVC": "Support Vector Machine",
        "LinearSVC": "Support Vector Machine",
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest",
        "XGBClassifier": "XGBoost",
        "CatBoostClassifier": "CatBoost",
        "LGBMClassifier": "LightGBM",
        "AdaBoostClassifier": "AdaBoost",
        "ExtraTreesClassifier": "Extra Trees",
        "DecisionTreeClassifier": "Decision Tree",
        "MLPClassifier": "Neural Network",
        "HistGradientBoostingClassifier": "Hist Gradient Boosting",
        "GradientBoostingClassifier": "Gradient Boosting",
        "KNeighborsClassifier": "K-Nearest Neighbors",
    }

    @staticmethod
    def _get_estimator(model):
        """Return the final estimator from a fitted sklearn Pipeline, if any."""
        if hasattr(model, "named_steps"):
            return model.named_steps.get("model", model)
        return model

    @classmethod
    def _get_model_name(cls, model):
        estimator = cls._get_estimator(model)
        estimator_class = type(estimator).__name__
        return cls.MODEL_DISPLAY_NAMES.get(estimator_class, estimator_class)

    @staticmethod
    def _load_model_comparison():
        """Load cross-validation comparisons written by ModelTrainer."""
        comparison_path = REPORTS_DIR / "model_comparison.csv"
        if not comparison_path.exists():
            logger.warning(
                "Model comparison file not found at %s; dashboard comparison will be empty.",
                comparison_path,
            )
            return []

        try:
            comparison_df = pd.read_csv(comparison_path)
            # Ensure NaN/NumPy scalar values are JSON-safe.
            comparison_df = comparison_df.replace({np.nan: None})
            return comparison_df.to_dict(orient="records")
        except Exception:
            logger.exception("Could not load model comparison CSV.")
            return []

    @staticmethod
    def _get_aligned_probabilities(model, X_holdout, model_classes, n_classes):
        """Align predict_proba columns to CLASS_NAMES' numeric class IDs."""
        if not hasattr(model, "predict_proba"):
            return None

        try:
            raw_probabilities = np.asarray(model.predict_proba(X_holdout), dtype=float)
        except Exception:
            logger.exception("Could not obtain prediction probabilities.")
            return None

        if raw_probabilities.ndim != 2:
            logger.warning("Model probabilities were not a 2D array.")
            return None

        if raw_probabilities.shape[1] != len(model_classes):
            logger.warning("Probability columns do not match model classes.")
            return None

        aligned = np.zeros((len(X_holdout), n_classes), dtype=float)
        for column_index, class_id in enumerate(model_classes):
            class_id = int(class_id)
            if 0 <= class_id < n_classes:
                aligned[:, class_id] = raw_probabilities[:, column_index]

        row_sums = aligned.sum(axis=1, keepdims=True)
        valid_rows = row_sums[:, 0] > 0
        aligned[valid_rows] /= row_sums[valid_rows]
        return aligned

    @staticmethod
    def _make_confidence_table(y_true, y_pred, confidence):
        thresholds = [0.6, 0.7, 0.8, 0.9]
        table = []
        total = len(y_true)

        for threshold in thresholds:
            selected = confidence >= threshold
            patients = int(selected.sum())
            accuracy = (
                float(accuracy_score(y_true[selected], y_pred[selected]))
                if patients
                else None
            )
            table.append({
                "threshold": threshold,
                "coverage": float(patients / total) if total else 0.0,
                "accuracy": accuracy,
                "patients": patients,
            })

        return table

    @staticmethod
    def _make_feature_importance(model, X_holdout, y_true):
        """Estimate feature importance as the drop in holdout accuracy when shuffled."""
        if not isinstance(X_holdout, pd.DataFrame) or X_holdout.empty:
            return []

        try:
            importance_result = permutation_importance(
                model,
                X_holdout,
                y_true,
                scoring="accuracy",
                n_repeats=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            rows = []
            for index, feature in enumerate(X_holdout.columns):
                importance = float(importance_result.importances_mean[index])
                rows.append({
                    "feature": str(feature),
                    "label": FEATURE_LABELS.get(str(feature), str(feature)),
                    "importance": importance,
                })
            rows.sort(key=lambda row: row["importance"], reverse=True)
            return rows[:12]
        except Exception:
            logger.exception("Feature importance calculation failed; using an empty list.")
            return []

    def initiate_model_evaluation(
        self,
        model,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
    ) -> ModelEvaluationArtifact:
        logger.info("========== MODEL EVALUATION STARTED ==========")

        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Normalize holdout target to the numeric class IDs used for training.
            y_true_raw = pd.Series(y_holdout).reset_index(drop=True)
            class_to_index = {
                str(name).strip(): index
                for index, name in enumerate(CLASS_NAMES)
            }

            if pd.api.types.is_numeric_dtype(y_true_raw):
                y_true = pd.to_numeric(y_true_raw, errors="raise").astype(int).to_numpy()
            else:
                mapped = y_true_raw.astype("string").str.strip().map(class_to_index)
                if mapped.isna().any():
                    unknown = sorted(y_true_raw[mapped.isna()].astype(str).unique())
                    raise ValueError(
                        f"Unknown target labels in holdout data: {unknown}. "
                        f"Expected labels: {list(CLASS_NAMES)}"
                    )
                y_true = mapped.astype(int).to_numpy()

            # Generate predictions from the fitted model/pipeline.
            y_pred = np.asarray(model.predict(X_holdout)).astype(int).ravel()
            if len(y_true) != len(y_pred):
                raise ValueError(
                    "The number of holdout labels does not match the number of predictions."
                )
            if len(y_true) == 0:
                raise ValueError("Holdout data is empty.")

            estimator = self._get_estimator(model)
            if not hasattr(estimator, "classes_"):
                raise ValueError("Unable to identify the fitted model's classes.")
            model_classes = np.asarray(estimator.classes_).astype(int)
            n_classes = len(CLASS_NAMES)
            class_ids = list(range(n_classes))

            if np.any(y_true < 0) or np.any(y_true >= n_classes):
                raise ValueError("Holdout numeric class IDs do not match CLASS_NAMES.")
            if np.any(y_pred < 0) or np.any(y_pred >= n_classes):
                raise ValueError("Predicted numeric class IDs do not match CLASS_NAMES.")
            if not set(np.unique(y_true)).issubset(set(model_classes.tolist())):
                raise ValueError(
                    "Holdout labels contain classes not seen by the model. "
                    f"Holdout: {sorted(set(y_true))}; model classes: {model_classes.tolist()}"
                )

            target_names = list(CLASS_NAMES)
            probabilities = self._get_aligned_probabilities(
                model, X_holdout, model_classes, n_classes
            )

            # Core metrics.
            accuracy = float(accuracy_score(y_true, y_pred))
            macro_precision = float(precision_score(
                y_true, y_pred, labels=class_ids, average="macro", zero_division=0
            ))
            macro_recall = float(recall_score(
                y_true, y_pred, labels=class_ids, average="macro", zero_division=0
            ))
            macro_f1 = float(f1_score(
                y_true, y_pred, labels=class_ids, average="macro", zero_division=0
            ))
            weighted_f1 = float(f1_score(
                y_true, y_pred, labels=class_ids, average="weighted", zero_division=0
            ))
            weighted_precision = float(precision_score(
                y_true, y_pred, labels=class_ids, average="weighted", zero_division=0
            ))
            weighted_recall = float(recall_score(
                y_true, y_pred, labels=class_ids, average="weighted", zero_division=0
            ))

            # Per-class metrics in the exact shape expected by the frontend.
            report_dict = classification_report(
                y_true,
                y_pred,
                labels=class_ids,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
            per_class = {}
            for name in target_names:
                row = report_dict[name]
                per_class[name] = {
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1-score"]),
                    "support": int(row["support"]),
                }

            cm = confusion_matrix(y_true, y_pred, labels=class_ids)
            counts = np.bincount(y_true, minlength=n_classes)
            baseline_accuracy = float(counts.max() / len(y_true))

            top2_accuracy = None
            confidence_table = []
            confidence = None
            auc_macro = None
            auc_per_class = {name: None for name in target_names}

            if probabilities is not None:
                confidence = probabilities.max(axis=1)
                top_k = min(2, n_classes)
                top2_class_ids = np.argsort(probabilities, axis=1)[:, -top_k:]
                top2_accuracy = float(
                    np.any(top2_class_ids == y_true[:, None], axis=1).mean()
                )
                confidence_table = self._make_confidence_table(
                    y_true, y_pred, confidence
                )

                # AUC can be undefined if the holdout set has no positive or
                # negative examples for a class. Keep undefined values as null.
                try:
                    if len(np.unique(y_true)) == 2:
                        positive_class = sorted(np.unique(y_true))[-1]
                        auc_macro = float(roc_auc_score(
                            (y_true == positive_class).astype(int),
                            probabilities[:, positive_class],
                        ))
                    elif len(np.unique(y_true)) == n_classes:
                        auc_macro = float(roc_auc_score(
                            y_true,
                            probabilities,
                            labels=class_ids,
                            multi_class="ovr",
                            average="macro",
                        ))
                    for class_id, name in enumerate(target_names):
                        binary_true = (y_true == class_id).astype(int)
                        if len(np.unique(binary_true)) == 2:
                            auc_per_class[name] = float(roc_auc_score(
                                binary_true, probabilities[:, class_id]
                            ))
                except Exception:
                    logger.exception("Could not calculate one or more ROC-AUC metrics.")

            # Keep the previous evaluation report for any existing consumer.
            legacy_metrics = {
                "Accuracy": accuracy,
                "Macro Precision": macro_precision,
                "Macro Recall": macro_recall,
                "Macro F1": macro_f1,
                "Weighted F1": weighted_f1,
                "ROC-AUC Macro": auc_macro,
                "ROC-AUC Per Class": auc_per_class,
                "Holdout Samples": int(len(y_true)),
                "Classes": target_names,
            }
            legacy_metrics_path = REPORTS_DIR / "evaluation_metrics.json"
            with open(legacy_metrics_path, "w", encoding="utf-8") as file:
                json.dump(legacy_metrics, file, indent=4, ensure_ascii=False)

            # The frontend reads this file at GET /api/metrics. Maintain its schema.
            model_name = self._get_model_name(model)
            confidence_threshold = float(CONFIDENCE_THRESHOLD)
            if probabilities is not None:
                matched_threshold = next(
                    (row for row in confidence_table
                     if row["threshold"] == confidence_threshold),
                    confidence_table[-1] if confidence_table else None,
                )
            else:
                matched_threshold = None

            dashboard_metrics = {
                "model_name": model_name,
                "holdout_size": int(len(y_true)),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "baseline_accuracy": baseline_accuracy,
                "top2_accuracy": top2_accuracy,
                "auc_macro": auc_macro,
                "auc_per_class": auc_per_class,
                "confidence_threshold": confidence_threshold,
                "confidence_table": confidence_table,
                "per_class": per_class,
                "confusion_matrix": cm.tolist(),
                "classes": target_names,
                "model_comparison": self._load_model_comparison(),
                "feature_importance": self._make_feature_importance(
                    model, X_holdout, y_true
                ),
                # Extra useful metrics that don't change current frontend fields.
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
            }

            with open(METRICS_FILE, "w", encoding="utf-8") as file:
                json.dump(dashboard_metrics, file, indent=2, ensure_ascii=False)

            # Save the detailed classification report and confusion matrix.
            classification_report_path = REPORTS_DIR / "classification_report.csv"
            pd.DataFrame(report_dict).T.to_csv(
                classification_report_path,
                index=True,
                index_label="Class",
            )

            confusion_matrix_path = REPORTS_DIR / "confusion_matrix.csv"
            pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
                confusion_matrix_path,
                index=True,
            )

            confusion_matrix_plot_path = REPORTS_DIR / "confusion_matrix.png"
            fig, ax = plt.subplots(figsize=(8, 6))
            image = ax.imshow(cm, interpolation="nearest")
            fig.colorbar(image, ax=ax)
            ax.set(
                xticks=np.arange(len(target_names)),
                yticks=np.arange(len(target_names)),
                xticklabels=target_names,
                yticklabels=target_names,
                xlabel="Predicted label",
                ylabel="Actual label",
                title="Confusion Matrix",
            )
            plt.setp(
                ax.get_xticklabels(),
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            threshold = cm.max() / 2 if cm.size and cm.max() else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > threshold else "black",
                    )
            fig.tight_layout()
            fig.savefig(confusion_matrix_plot_path, dpi=150)
            plt.close(fig)

            logger.info("Model evaluated: %s", model_name)
            logger.info("Test accuracy: %.4f", accuracy)
            logger.info("Macro F1: %.4f", macro_f1)
            logger.info("Dashboard metrics saved: %s", METRICS_FILE)
            logger.info("Evaluation metrics saved: %s", legacy_metrics_path)
            logger.info("Classification report saved: %s", classification_report_path)
            logger.info("Confusion matrix saved: %s", confusion_matrix_path)
            logger.info("Confusion matrix plot saved: %s", confusion_matrix_plot_path)
            logger.info("========== MODEL EVALUATION COMPLETED ==========")

            return ModelEvaluationArtifact(
                evaluation_metrics_path=str(legacy_metrics_path),
                classification_report_path=str(classification_report_path),
                confusion_matrix_path=str(confusion_matrix_path),
                confusion_matrix_plot_path=str(confusion_matrix_plot_path),
                metrics=legacy_metrics,
            )

        except Exception as e:
            logger.exception("Error occurred during model evaluation.")
            raise AGDPMException(e, None) from e
