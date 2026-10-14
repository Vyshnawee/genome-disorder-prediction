from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"            # original file from Kaggle, never modified
WORKING_DATA_DIR = DATA_DIR / "working"    # copy used by the pipeline
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "genetic_disease_dataset_simple_classification.csv"
DATA_FILE = WORKING_DATA_DIR / "genetic_disease_dataset_simple_classification.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VALIDATION_DIR = ARTIFACTS_DIR / "validation"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

MODEL_FILE = MODEL_DIR / "model_pipeline.pkl"
MODEL_INFO_FILE = MODEL_DIR / "model_info.json"
METRICS_FILE = REPORTS_DIR / "metrics.json"

LOG_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------
# Dataset: "Genetic Disease Prediction Dataset" (Kaggle,
# syeddanish5/genetic-disease-prediction-dataset, CC-BY-SA-4.0)
# ---------------------------------------------------------

TARGET_COLUMN = "Genetic_Disorder_Classification"
ID_COLUMN = "Patient Id"   # optional in uploaded CSVs

# Label order as documented by the dataset author (0..4)
CLASS_NAMES = [
     "Single gene inherited disease",
     "Multifactorial genetic disorder"
#     "Thalassemia",
#     "Hemophilia",
#     "Breast Cancer (BRCA1/p53)",
#     "Sickle Cell Anemia",
#     "Cystic Fibrosis",
]

CLASS_INFO = {
    "Single gene inherited disease": (
        "A genetic disorder primarily caused by a mutation in a single gene. "
        "Examples include Thalassemia, Hemophilia, Sickle Cell Anemia, "
        "and Cystic Fibrosis."
    ),

    "Multifactorial genetic disorder": (
        "A disorder influenced by multiple genetic factors and environmental "
        "factors. Breast cancer can be multifactorial, although some hereditary "
        "cases are caused by mutations in a single gene such as BRCA1 or BRCA2."
    ),
}

# CLASS_INFO = {
#     "Thalassemia": "Inherited blood disorder in which the body makes too little hemoglobin. Typical signs are low hemoglobin and high fetal hemoglobin.",
#     "Hemophilia": "Inherited bleeding disorder in which blood does not clot properly, caused by missing clotting factors (VIII or IX).",
#     "Breast Cancer (BRCA1/p53)": "Hereditary breast cancer linked to BRCA1 and TP53 (p53) gene changes. Typical signs are high BRCA1 expression and p53 mutation.",
#     "Sickle Cell Anemia": "Inherited disorder of the HBB gene in which red blood cells become sickle-shaped. Typical sign is a high percentage of sickled red cells.",
#     "Cystic Fibrosis": "Inherited disorder of the CFTR gene that affects the lungs and digestive system. Typical sign is high sweat chloride (60 mmol/L or more).",
# }

# ---------------------------------------------------------
# Features (exactly what a patient enters in the UI)
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "Age",
    "Hemoglobin",
    "Fetal_Hemoglobin",
    "RDW_CV",
    "Serum_Ferritin",
    "BRCA1_Expression",
    "Sweat_Chloride",
    "Sickled_RBC_Percent",
    "IL6_Level",
]

# 0/1 features, shown in the UI as two named choices
BINARY_FEATURES = {
    "Gender": ["Male", "Female"],          # 0 = Male, 1 = Female
    "Family_History": ["No", "Yes"],       # 0 = No,   1 = Yes
    "p53_Mutation": ["Normal", "Mutated"], # 0 = Normal, 1 = Mutated
}

# INPUT_FEATURES = ["Age", "Gender", "Family_History"] + NUMERIC_FEATURES[1:6] + ["p53_Mutation"] + NUMERIC_FEATURES[6:]



INPUT_FEATURES = [
    "Age",
    "Gender",
    "Family_History",
    "Hemoglobin",
    "Fetal_Hemoglobin",
    "RDW_CV",
    "Serum_Ferritin",
    "BRCA1_Expression",
    "IL6_Level",
]

# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
CV_FOLDS = 5

CONFIDENCE_THRESHOLD = 0.80
CONFIDENCE_THRESHOLDS_TO_REPORT = [0.60, 0.70, 0.80, 0.90]

# ---------------------------------------------------------
# UI: labels, units, normal ranges and form sections
# ---------------------------------------------------------

FEATURE_LABELS = {
    "Age": "Age (years)",
    "Gender": "Gender",
    "Family_History": "Family history of genetic disease",
    "Hemoglobin": "Hemoglobin (g/dL)",
    "Fetal_Hemoglobin": "Fetal hemoglobin, HbF (%)",
    "RDW_CV": "RDW-CV (%)",
    "Serum_Ferritin": "Serum ferritin (ng/mL)",
    "BRCA1_Expression": "BRCA1 expression (0–1)",
    "p53_Mutation": "p53 (TP53) gene",
    "Sweat_Chloride": "Sweat chloride (mmol/L)",
    "Sickled_RBC_Percent": "Sickled red blood cells (%)",
    "IL6_Level": "IL-6 level (pg/mL)",
}

# General adult reference values, shown as hints in the form
NORMAL_RANGES = {
    "Hemoglobin": "Normal ≈ 12–17",
    "Fetal_Hemoglobin": "Normal < 2",
    "RDW_CV": "Normal ≈ 11.5–14.5",
    "Serum_Ferritin": "Normal ≈ 20–250",
    "BRCA1_Expression": "Higher in hereditary breast cancer",
    "Sweat_Chloride": "Normal < 30 · CF ≥ 60",
    "Sickled_RBC_Percent": "Normal ≈ 0",
    "IL6_Level": "Normal < 7",
}

FORM_SECTIONS = [
    {"title": "Patient Information", "fields": ["Age", "Gender", "Family_History"]},
    {"title": "Blood Tests", "fields": ["Hemoglobin", "Fetal_Hemoglobin", "RDW_CV", "Serum_Ferritin", "Sickled_RBC_Percent"]},
    {"title": "Genetic Markers", "fields": ["BRCA1_Expression", "p53_Mutation"]},
    {"title": "Other Tests", "fields": ["Sweat_Chloride", "IL6_Level"]},
]
