import io
import json
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import (
    BINARY_FEATURES,
    CLASS_NAMES,
    ID_COLUMN,
    INPUT_FEATURES,
    METRICS_FILE,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
)
from src.pipeline.predict_pipeline import PredictPipeline


STATIC_DIR = PROJECT_ROOT / "app" / "static"
MAX_BATCH_ROWS = 20000
ANSWER_COLUMNS = ["Actual Disease", TARGET_COLUMN]

app = FastAPI(title="Genetic Disease Predictor API", version="3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predictor = PredictPipeline()


def validate_patient(patient: dict[str, Any]) -> dict[str, Any]:
    """Keep known fields only and reject values the model cannot use."""
    clean = {}
    errors = []

    for column in INPUT_FEATURES:
        value = patient.get(column)

        if value is None or value == "":
            clean[column] = None
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{column}' must be a number.")
            continue

        if column in BINARY_FEATURES and value not in (0, 1):
            errors.append(f"'{column}' must be 0 or 1.")
        elif column in NUMERIC_FEATURES and value < 0:
            errors.append(f"'{column}' cannot be negative.")

        clean[column] = value

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return clean


def read_answers(df: pd.DataFrame) -> list | None:
    """If the CSV contains the true disease, return it as class names."""
    for column in ANSWER_COLUMNS:
        if column in df.columns:
            values = df[column]
            if pd.api.types.is_numeric_dtype(values):
                values = values.map(lambda v: CLASS_NAMES[int(v)] if pd.notna(v) and 0 <= v < len(CLASS_NAMES) else None)
            return values.where(values.isin(CLASS_NAMES), None).tolist()
    return None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": predictor.model_info["model_name"]}


@app.get("/api/model-info")
def model_info():
    return predictor.model_info


@app.get("/api/metrics")
def metrics():
    with open(METRICS_FILE, encoding="utf-8") as file:
        return json.load(file)


@app.get("/api/sample")
def sample_patient():
    """A random patient from the holdout set (not used in training), for demos."""
    X = pd.read_csv(PROCESSED_DATA_DIR / "X_holdout_clean.csv")
    y = pd.read_csv(PROCESSED_DATA_DIR / "y_holdout.csv").squeeze("columns")

    index = int(pd.Series(range(len(X))).sample(1).iloc[0])
    row = X.iloc[index].astype(object).where(X.iloc[index].notna(), None).to_dict()

    return {"patient": row, "actual": y.iloc[index]}


@app.post("/api/predict")
def predict(patient: dict[str, Any] = Body(...)):
    return predictor.predict_one(validate_patient(patient))


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the CSV file.")

    df = df.dropna(how="all")

    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV file has no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_BATCH_ROWS} rows per file.")

    matched = [column for column in INPUT_FEATURES if column in df.columns]
    if not matched:
        raise HTTPException(
            status_code=400,
            detail="No known feature columns found. Use the same column names as the dataset.",
        )

    predictions = predictor.predict_dataframe(df)
    ids = df[ID_COLUMN].tolist() if ID_COLUMN in df.columns else list(range(1, len(df) + 1))
    answers = read_answers(df)

    results = []
    for position, (patient_id, prediction) in enumerate(zip(ids, predictions)):
        row = {"id": str(patient_id), **prediction}
        if answers is not None:
            row["actual"] = answers[position]
        results.append(row)

    response = {
        "rows": len(predictions),
        "matched_columns": len(matched),
        "total_columns": len(INPUT_FEATURES),
        "results": results,
    }

    # When the file contains the true disease, report how many were correct
    if answers is not None:
        known = [r for r in results if r["actual"]]
        if known:
            response["accuracy"] = sum(r["actual"] == r["prediction"] for r in known) / len(known)
            response["checked_rows"] = len(known)

    return response
