
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


def print_summary(model_name, cv_results, metrics):
    print("\n" + "=" * 64)
    print("MODEL COMPARISON (Cross-validation on training split)")
    print("=" * 64)
    print(
        cv_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 64)
    print(
        f"FINAL RESULTS ON HOLDOUT SET "
        f"({metrics['Holdout Samples']} unseen patients)"
    )
    print("=" * 64)
    print(f"Best model           : {model_name}")

    metric_keys = [
        ("Accuracy", "Accuracy"),
        ("Macro Precision", "Macro Precision"),
        ("Macro Recall", "Macro Recall"),
        ("Macro F1", "Macro F1"),
        ("Weighted F1", "Weighted F1"),
        ("ROC-AUC Macro", "ROC-AUC Macro"),
    ]

    for label, key in metric_keys:
        value = metrics.get(key)

        if value is None:
            print(f"{label:<22}: N/A")
        elif isinstance(value, (int, float)):
            print(f"{label:<22}: {value:.4f}")
        else:
            print(f"{label:<22}: {value}")

    print("\nPer-class ROC-AUC:")
    for name, score in metrics.get("ROC-AUC Per Class", {}).items():
        if score is None:
            print(f"  {name:<35}: N/A")
        else:
            print(f"  {name:<35}: {score:.4f}")

    print("=" * 64)


if __name__ == "__main__":

    print("1/5 Data ingestion ...")
    data = DataIngestion().initiate_data_ingestion().data

    print("2/5 Data validation ...")
    DataValidation().initiate_data_validation(data)

    print("3/5 Data preprocessing ...")
    splits = DataPreprocessing().initiate_data_preprocessing(data)

    print("4/5 Model training ...")
    trainer_artifact = ModelTrainer().initiate_model_training(
        splits.X_train,
        splits.y_train,
    )

    print("5/5 Model evaluation ...")
    evaluation_artifact = ModelEvaluation().initiate_model_evaluation(
        trainer_artifact.model,
        splits.X_holdout,
        splits.y_holdout,
    )

    print_summary(
        trainer_artifact.model_name,
        trainer_artifact.cv_results,
        evaluation_artifact.metrics,
    )

    print("\nDone. Start the app with: uvicorn app.main:app --reload")