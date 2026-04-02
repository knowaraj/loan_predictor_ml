import joblib
import numpy as np
from django.conf import settings
from pathlib import Path

# ==============================
# GLOBAL CACHES
# ==============================
MODEL = None
EXPLAINER = None
FEATURE_NAMES = None

# ==============================
# PATHS
# ==============================
BASE = Path(settings.BASE_DIR) / "ml" / "models"

MODEL_PATH = BASE / "pipeline_xgb.joblib"
EXPLAINER_PATH = BASE / "shap_explainer.joblib"
FEATURE_PATH = BASE / "feature_names.joblib"


# ==============================
# LOAD MODEL (Pipeline)
# ==============================
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = joblib.load(MODEL_PATH)
    return MODEL


# ==============================
# LOAD SHAP EXPLAINER
# ==============================
def get_explainer():
    global EXPLAINER
    if EXPLAINER is None:
        EXPLAINER = joblib.load(EXPLAINER_PATH)
    return EXPLAINER


# ==============================
# LOAD FEATURE NAMES
# ==============================
def get_feature_names():
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = joblib.load(FEATURE_PATH)
    return FEATURE_NAMES


# ==============================
# MAIN PREDICTION FUNCTION
# ==============================
def predict_with_explanations(input_dict):
    """
    input_dict = {
        "Gender": "...",
        "Married": "...",
        "ApplicantIncome": 5000,
        ...
    }
    """

    model = get_model()
    explainer = get_explainer()
    feature_names = get_feature_names()

    # Convert input into a DataFrame
    import pandas as pd
    df = pd.DataFrame([input_dict])

    # Run through pipeline → get prediction & proba
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    # SHAP requires transformed features
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(df)

    # Compute SHAP values
    shap_values = explainer.shap_values(transformed)[0]

    # Combine SHAP values with feature names
    shap_dict = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_reasons = shap_dict[:5]  # top 5 contributions

    return {
        "prediction": int(pred),
        "probability": float(proba),
        "top_reasons": top_reasons
    }
