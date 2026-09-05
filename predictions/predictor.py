"""
Loan Prediction Module

This module handles ML model inference, explainability, and prediction generation
using XGBoost pipeline with SHAP-based feature importance.

Author: Loan Predictor Team
Version: 1.0
"""

import joblib
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from django.conf import settings
from pathlib import Path

logger = logging.getLogger(__name__)

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
def get_model() -> Any:
    """
    Load and cache the XGBoost prediction pipeline.
    
    Uses lazy loading - loads only on first call and caches for subsequent calls.
    This improves performance by avoiding repeated disk I/O.
    
    Returns:
        Any: Loaded scikit-learn pipeline containing preprocessor and XGBoost model.
        
    Raises:
        FileNotFoundError: If model file not found at MODEL_PATH.
    """
    global MODEL
    if MODEL is None:
        logger.info(f"Loading ML model from {MODEL_PATH}")
        MODEL = joblib.load(MODEL_PATH)
        logger.info("ML model loaded successfully")
    return MODEL


# ==============================
# LOAD SHAP EXPLAINER
# ==============================
def get_explainer() -> Any:
    """
    Load and cache the SHAP explainer for feature importance calculation.
    
    The explainer is used to compute SHAP values which provide interpretable
    explanations for individual predictions.
    
    Returns:
        Any: Loaded SHAP explainer instance.
        
    Raises:
        FileNotFoundError: If explainer file not found at EXPLAINER_PATH.
    """
    global EXPLAINER
    if EXPLAINER is None:
        logger.info(f"Loading SHAP explainer from {EXPLAINER_PATH}")
        EXPLAINER = joblib.load(EXPLAINER_PATH)
        logger.info("SHAP explainer loaded successfully")
    return EXPLAINER


# ==============================
# LOAD FEATURE NAMES
# ==============================
def get_feature_names() -> List[str]:
    """
    Load and cache the feature names list for the ML model.
    
    Feature names are required to map SHAP values to their corresponding features.
    
    Returns:
        List[str]: List of feature names used by the model.
        
    Raises:
        FileNotFoundError: If feature names file not found at FEATURE_PATH.
    """
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        logger.info(f"Loading feature names from {FEATURE_PATH}")
        FEATURE_NAMES = joblib.load(FEATURE_PATH)
        logger.info(f"Loaded {len(FEATURE_NAMES)} features")
    return FEATURE_NAMES


# ==============================
# MAIN PREDICTION FUNCTION
# ==============================
def predict_with_explanations(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate loan default prediction with SHAP-based explanations.
    
    This function takes borrower information, runs it through the ML pipeline,
    and returns prediction along with top 5 most influential features.
    
    Args:
        input_dict (Dict[str, Any]): Dictionary containing borrower features.
            Expected keys: Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
            Credit_History, Property_Area
            
    Returns:
        Dict[str, Any]: Dictionary containing:
            - prediction (int): 1 for likely default, 0 for likely repayment
            - probability (float): Probability of default (0.0 to 1.0)
            - top_reasons (List[Tuple]): Top 5 features by SHAP impact
              Format: [(feature_name, shap_value), ...]
              
    Example:
        >>> result = predict_with_explanations({
        ...     "Gender": "Male",
        ...     "Married": "Yes",
        ...     "ApplicantIncome": 5000,
        ...     # ... other fields
        ... })
        >>> print(f"Default probability: {result['probability']:.2%}")
    """
    try:
        model = get_model()
        explainer = get_explainer()
        feature_names = get_feature_names()
        
        # Convert input into a DataFrame
        df = pd.DataFrame([input_dict])
        logger.info(f"Processing prediction for input with {len(df.columns)} features")
        
        # Run through pipeline → get prediction & proba
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]
        logger.info(f"Model prediction: {pred}, probability: {proba:.4f}")
        
        # SHAP requires transformed features from pipeline
        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(df)
        
        # Compute SHAP values for explanation
        shap_values = explainer.shap_values(transformed)[0]
        
        # Combine SHAP values with feature names and sort by absolute impact
        shap_dict = sorted(
            zip(feature_names, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_reasons = shap_dict[:5]  # top 5 contributions
        logger.info(f"Top 5 reasons identified: {[f[0] for f in top_reasons]}")
        
        return {
            "prediction": int(pred),
            "probability": float(proba),
            "top_reasons": top_reasons
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise
