# GeneScope, Version 3 (Master): Genetic Disease Predictor

Predicts which of five genetic diseases a patient most likely has
(Thalassemia, Hemophilia, Breast Cancer (BRCA1/p53), Sickle Cell Anemia,
Cystic Fibrosis) from lab results and genetic markers, with a web UI for
single-patient and CSV batch predictions.

## Results (200 held-back patients the model never saw)

| Metric | Value |
|---|---|
| Accuracy | 95.5% |
| Macro precision / recall / F1 | 95.6% / 95.5% / 0.955 |
| Top-2 accuracy | 97.5% |
| AUC (macro) | 0.974 |
| Guessing the most common class | 20.5% |

Train accuracy (95.5%) is the same as test accuracy, so the model is not memorising.

## Dataset

*Genetic Disease Prediction Dataset* by syeddanish5 on Kaggle
(`syeddanish5/genetic-disease-prediction-dataset`, license CC-BY-SA-4.0).
1,000 patients, 12 features, 5 balanced classes. The author describes it as
"synthetic yet realistic". Every patient has one of the five diseases, so
the model cannot predict "healthy".

To download it again:

```
kaggle datasets download syeddanish5/genetic-disease-prediction-dataset -p data/raw --unzip
```

Other Kaggle datasets were checked and rejected: Huntington Disease and
Cancer Variant had random labels (accuracy equal to guessing), and Human DNA
sequences predict gene families, not patients.

## Run

```
venv\Scripts\activate
pip install -r requirements.txt
python main.py                      # validate, clean, train, evaluate (~1 min)
uvicorn app.main:app --reload       # start the app
```

Open http://127.0.0.1:8000

To check accuracy yourself: on the **Batch** page upload
`data/processed/holdout_patients_with_answers.csv`. The app compares its
predictions with the true diseases in the file.

## Pipeline

1. **Ingestion**: copy `data/raw` → `data/working` and read the copy.
2. **Validation**: report in `artifacts/validation/validation_report.txt`.
3. **Preprocessing**: drop empty and duplicate rows, map labels 0–4 to disease
   names, clip impossible negative lab values to 0, then a stratified 80/20 split.
4. **Training**: Logistic Regression, SVM, Random Forest, Hist Gradient
   Boosting and XGBoost compared with 5-fold CV. The lowest log loss is kept.
5. **Evaluation**: holdout only. Accuracy, precision, recall, F1, top-2, AUC,
   accuracy by confidence, confusion matrix and feature importance.

> Educational project. Not for clinical use.
