from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    """
    Imputation + scaling, packaged as one sklearn pipeline so it is
    saved together with the model. Input: the output of clean_features().

    All inputs are numeric lab values or 0/1 flags, and each disease has
    its own clear marker, so no extra engineered features are needed.
    """
    return Pipeline(steps=[
        # add_indicator=True keeps the information "this test was not done"
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
