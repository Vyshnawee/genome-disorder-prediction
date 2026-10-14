import json

import joblib
import pandas as pd

from src.components.data_preprocessing import clean_features
from src.config import CLASS_NAMES, MODEL_FILE, MODEL_INFO_FILE


class PredictPipeline:
    """
    Loads the saved model once and predicts for one patient or many.
    Input uses the original dataset column names.
    """

    def __init__(self):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_FILE}. Run 'python main.py' first."
            )

        self.model = joblib.load(MODEL_FILE)

        with open(MODEL_INFO_FILE, encoding="utf-8") as file:
            self.model_info = json.load(file)

        self.threshold = self.model_info["confidence_threshold"]

    def _format(self, probabilities) -> dict:
        ranked = sorted(
            zip(CLASS_NAMES, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        prediction, confidence = ranked[0]

        if confidence >= self.threshold:
            level = "High"
        elif confidence >= 0.5:
            level = "Moderate"
        else:
            level = "Low"

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "confidence_level": level,
            "is_confident": bool(confidence >= self.threshold),
            "top2": [name for name, _ in ranked[:2]],
            "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities)},
        }

    def predict_dataframe(self, df: pd.DataFrame) -> list[dict]:
        features = clean_features(df)
        return [self._format(row) for row in self.model.predict_proba(features)]

    def predict_one(self, patient: dict) -> dict:
        return self.predict_dataframe(pd.DataFrame([patient]))[0]
