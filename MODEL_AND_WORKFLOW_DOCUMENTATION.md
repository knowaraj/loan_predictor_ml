# Comprehensive Model & Project Workflow Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [ML Model Details](#ml-model-details)
4. [Complete Project Workflow](#complete-project-workflow)
5. [Data Flow & Transformation](#data-flow--transformation)
6. [Feature Engineering](#feature-engineering)
7. [Model Inference Pipeline](#model-inference-pipeline)
8. [Explainability (SHAP)](#explainability-shap)
9. [System Components](#system-components)
10. [Database Schema](#database-schema)
11. [API Integration Points](#api-integration-points)
12. [Security & Authentication](#security--authentication)

---

## Executive Summary

The **Loan Predictor** is an end-to-end machine learning application that predicts loan default risk using **XGBoost** with **SHAP** explainability. The system combines a Django backend, React frontend, and ML pipeline to enable users to submit loan applications and receive AI-driven predictions with interpretable explanations.

**Key Statistics:**
- Model: XGBoost (Gradient Boosting)
- Trees: 300 estimators
- Features: 11 (5 categorical, 6 numerical)
- Target: Binary classification (Loan Default: Yes/No)
- Explainability: SHAP (SHapley Additive exPlanations)

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│  ┌──────────────────┐           ┌──────────────────┐            │
│  │  Loan Users      │           │  Admin Users     │            │
│  │  (Applicants)    │           │  (Reviewers)     │            │
│  └────────┬─────────┘           └────────┬─────────┘            │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
            │ HTTP/REST                    │ HTTP/REST
┌───────────▼──────────────────────────────▼──────────────────────┐
│                      PRESENTATION LAYER                         │
│  ┌────────────────────────────┐  ┌──────────────────────────┐   │
│  │  React Frontend (Vite)     │  │  Django Admin Templates  │   │
│  │  ├─ Loan Prediction Form   │  │  ├─ Dashboard           │   │
│  │  ├─ Loan History View      │  │  ├─ Loan Review         │   │
│  │  └─ Result Display         │  │  └─ User Management     │   │
│  └────────────────────────────┘  └──────────────────────────┘   │
└───────────┬──────────────────────────────┬──────────────────────┘
            │                              │
            │ Django Views                 │ Django Views
┌───────────▼──────────────────────────────▼──────────────────────┐
│                    APPLICATION LAYER                            │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │  predictions/        │  │  admin_panel/                    │ │
│  │  ├─ views.py         │  │  ├─ views.py                     │ │
│  │  ├─ models.py        │  │  ├─ models.py                    │ │
│  │  ├─ forms.py         │  │  ├─ forms.py                     │ │
│  │  ├─ urls.py          │  │  └─ urls.py                      │ │
│  │  └─ predictor.py ────┼──┼─→ ML INFERENCE ENGINE            │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
└───────────┬──────────────────────────────┬──────────────────────┘
            │                              │
            │ ORM Queries                  │ ORM Queries
┌───────────▼──────────────────────────────▼──────────────────────┐
│                      DATA LAYER                                 │
│  ┌──────────────────────┐         ┌──────────────────────┐      │
│  │   SQLite Database    │         │   ML Models (Joblib) │      │
│  │  ├─ Users            │         │  ├─ pipeline_xgb     │      │
│  │  ├─ LoanApplication  │         │  ├─ shap_explainer   │      │
│  │  ├─ PredictionResult │         │  └─ feature_names    │      │
│  │  └─ UserProfile      │         └──────────────────────┘      │
│  └──────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────┘
```

### Technology Stack Layers

**Layer 1: Frontend**
- React 18 with Vite bundler
- HTML5 form validation
- CSS3 responsive styling
- JavaScript ES6+

**Layer 2: Backend**
- Django 5.2 web framework
- SQLite database (development)
- Django ORM for data access
- RESTful view handlers

**Layer 3: ML Pipeline**
- XGBoost for classification
- Scikit-learn preprocessing
- SHAP for explainability
- Joblib for model serialization

---

## ML Model Details

### Model Specification

**Algorithm:** XGBoost (eXtreme Gradient Boosting)

**Hyperparameters:**
```python
{
    'n_estimators': 300,           # Number of boosting rounds
    'learning_rate': 0.05,         # Shrinkage (eta)
    'max_depth': 5,                # Maximum tree depth
    'min_samples_leaf': 8,         # Min samples in leaf node
    'subsample': 0.8,              # Fraction of samples for each tree
    'colsample_bytree': 0.8,       # Fraction of features for each tree
    'lambda': 1.0,                 # L2 regularization
    'gamma': 0.0,                  # Min loss reduction to split
    'min_child_weight': 1.0,       # Min sum of weights in leaf
    'early_stopping_rounds': 20    # Rounds without improvement before stop
}
```

**Model Type:** Binary Classification

**Target Variable:**
- **Name:** `Loan_Status` or `predicted_default`
- **Values:** 0 (Non-defaulter/Approved) or 1 (Defaulter/Rejected)
- **Probability Output:** 0.0 to 1.0

**Decision Threshold:** 0.65 (configurable via `PredictionConfig.threshold`)
- Probability > 0.65 → Classified as DEFAULTER
- Probability ≤ 0.65 → Classified as NON-DEFAULTER

### Model Output

```python
Prediction Output = {
    "prediction": int (0 or 1),           # Class label
    "probability": float (0.0-1.0),       # Probability of defaulting
    "top_reasons": [                      # Top 5 SHAP features
        ("feature_name", shap_value),
        ("feature_name", shap_value),
        ...
    ]
}
```

### Model Performance Metrics (Reference)

Typically evaluated on:
- **Accuracy:** Overall correctness
- **ROC-AUC:** Ranking ability
- **Precision:** False positive rate (important for loan approval)
- **Recall:** False negative rate (important for risk)
- **F1-Score:** Harmonic mean of precision and recall

---

## Complete Project Workflow

### 1. User Registration & Onboarding Flow

```
┌─────────────────────────────────────────────────────┐
│ NEW USER REGISTRATION FLOW                          │
└─────────────────────────────────────────────────────┘

Step 1: User visits registration page
        ↓
Step 2: User fills registration form
        ├─ Username (unique)
        ├─ Password (hashed)
        ├─ Email
        ├─ First/Last Name
        ├─ Mobile Number
        └─ Address
        ↓
Step 3: Form validation
        ├─ Server-side validation
        ├─ Password strength check
        └─ Uniqueness verification
        ↓
Step 4: User creation (atomic transaction)
        ├─ Create User (Django Auth)
        ├─ Create UserProfile (extended fields)
        └─ Generate success message
        ↓
Step 5: Redirect to login
```

**Database Changes:**
```sql
INSERT INTO auth_user (username, password, email, first_name, last_name, ...);
INSERT INTO predictions_userprofile (user_id, mobile_number, address);
```

### 2. Loan Application & Prediction Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ LOAN PREDICTION REQUEST FLOW                                     │
└──────────────────────────────────────────────────────────────────┘

PHASE 1: USER SUBMISSION
┌─────────────────────────────────────────┐
│ User fills loan prediction form         │
│                                         │
│ Fields Collected:                       │
│ • Demographics (gender, married, deps)  │
│ • Employment (self-employed status)     │
│ • Education level                       │
│ • Financial (incomes, loan amount, term)│
│ • Credit history                        │
│ • Property area                         │
└────────────┬────────────────────────────┘
             │ Form POST to /predict/
             ↓
┌─────────────────────────────────────────────────────┐
PHASE 2: FORM VALIDATION & DATA CAPTURE
│                                                       │
│ 1. Django receives POST request                      │
│ 2. CSRF token validation                            │
│ 3. Form validation (LoanApplicationForm)            │
│    - Field type checking                            │
│    - Required field validation                      │
│    - Income range validation                        │
│ 4. Save to database: LoanApplication(user=user)    │
│    - Capture all 11 input fields                    │
│    - Set created_at timestamp                       │
│    - Link to current user                           │
│                                                     │
└────────────┬────────────────────────────────────────┘
             │ DB: INSERT LoanApplication
             ↓
┌─────────────────────────────────────────────────────┐
PHASE 3: FEATURE PREPARATION
│                                                       │
│ Convert form input to ML-ready format:              │
│                                                     │
│ Input Dict = {                                      │
│   'Gender': 'Male',                 ← categorical   │
│   'Married': 'Yes',                 ← categorical   │
│   'Dependents': '1',                ← categorical   │
│   'Education': 'Graduate',          ← categorical   │
│   'Self_Employed': 'No',            ← categorical   │
│   'ApplicantIncome': 5000,          ← numerical     │
│   'CoapplicantIncome': 2000,        ← numerical     │
│   'LoanAmount': 120,                ← numerical     │
│   'Loan_Amount_Term': 360,          ← numerical     │
│   'Credit_History': 1,              ← numerical     │
│   'Property_Area': 'Urban'          ← categorical   │
│ }                                                    │
│                                                     │
│ Convert to Pandas DataFrame:                        │
│ df = pd.DataFrame([input_dict])                     │
│                                                     │
└────────────┬────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────┐
PHASE 4: ML PREDICTION
│                                                       │
│ 1. Load ML Pipeline (cached in memory)              │
│    └─ Contains: Preprocessor + XGBoost Model        │
│                                                     │
│ 2. Run through preprocessing:                       │
│    df_transformed = preprocessor.transform(df)      │
│                                                     │
│    Preprocessing steps:                             │
│    ├─ OneHotEncoder for categorical features       │
│    ├─ StandardScaler for numerical features        │
│    └─ SimpleImputer for missing values             │
│                                                     │
│ 3. Get raw predictions:                             │
│    pred_class = model.predict(df_transformed)      │
│    pred_proba = model.predict_proba(df_transformed)│
│                                                     │
│    Output: pred_class ∈ {0, 1}                     │
│            pred_proba ∈ [0.0, 1.0]                │
│                                                     │
│ 4. Apply threshold (from PredictionConfig):        │
│    if pred_proba > threshold:                      │
│        predicted = 1 (DEFAULTER)                   │
│    else:                                            │
│        predicted = 0 (NON-DEFAULTER)              │
│                                                     │
│    Default threshold = 0.65                         │
│                                                     │
└────────────┬────────────────────────────────────────┘
             │ Prediction complete
             ↓
┌─────────────────────────────────────────────────────┐
PHASE 5: EXPLAINABILITY (SHAP)
│                                                       │
│ 1. Load SHAP explainer (cached)                      │
│                                                     │
│ 2. Compute SHAP values:                             │
│    shap_values = explainer.shap_values(              │
│        df_transformed                               │
│    )                                                 │
│                                                     │
│    SHAP measures feature contribution to            │
│    prediction for THIS SPECIFIC applicant           │
│                                                     │
│ 3. Map SHAP values to feature names:                │
│    shap_dict = {                                    │
│        'ApplicantIncome': 0.15,    ← positive       │
│        'LoanAmount': -0.08,        ← negative       │
│        'Credit_History': 0.22,     ← strongest      │
│        ...                                          │
│    }                                                 │
│                                                     │
│ 4. Sort by absolute impact:                         │
│    top_5_reasons = sorted by |shap_value|          │
│                                                     │
│    Result: [(feature, impact), ...]                │
│                                                     │
└────────────┬────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────┐
PHASE 6: SAVE PREDICTION RESULT
│                                                       │
│ Create PredictionResult record:                      │
│                                                     │
│ pred = PredictionResult.objects.create(             │
│     application=loan_app,                           │
│     predicted_default=predicted,                    │
│     probability=float(pred_proba),                  │
│     model_version='v1'                              │
│ )                                                    │
│                                                     │
│ DB: INSERT PredictionResult                         │
│                                                     │
│ This creates audit trail of all predictions         │
│                                                     │
└────────────┬────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────┐
PHASE 7: DISPLAY RESULT TO USER
│                                                       │
│ Render result.html template with:                   │
│                                                     │
│ {                                                   │
│     'prediction': {                                 │
│         'id': 42,                                   │
│         'predicted_default': True/False             │
│         'probability': 0.73,                        │
│         'application': {                            │
│             'id': 15,                               │
│             'applicant_income': 5000,               │
│             'loan_amount': 120,                     │
│             ...                                     │
│         }                                            │
│     },                                              │
│     'top_reasons': [top 5 SHAP features]           │
│ }                                                    │
│                                                     │
│ Display to user:                                    │
│ ✓ Prediction outcome (Default/Non-default)         │
│ ✓ Confidence percentage                             │
│ ✓ Top 5 most influential factors (SHAP)            │
│ ✓ Option to apply for formal loan if approved      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 3. Formal Loan Application Flow

```
┌──────────────────────────────────────────────┐
│ FORMAL LOAN APPLICATION                      │
└──────────────────────────────────────────────┘

User PREDICTION (Above)
        ↓
User sees result (Default: True/False)
        ↓
If prediction == Non-defaulter:
    ↓
    User clicks "Apply for Loan" button
        ↓
    Fills formal application:
    ├─ Loan amount
    ├─ Loan tenure (months)
    └─ Loan purpose
        ↓
    Create LoanApplicationFormData record:
    - Status: 'Pending'
    - linked to PredictionResult
        ↓
ADMIN REVIEW PHASE:
    ↓
    Admin sees "Pending" loans in dashboard
        ↓
    Admin reviews application:
    ├─ View prediction details
    ├─ View user profile
    ├─ View financial summary
        ↓
    Admin makes decision:
    ├─ Approve → Status = 'Approved'
    ├─ Reject  → Status = 'Rejected'
    └─ Add remarks/comments
        ↓
    LoanApplicationFormData updated in DB
        ↓
User receives notification in loan history
```

### 4. Admin Dashboard Flow

```
┌──────────────────────────────────────────────┐
│ ADMIN DASHBOARD WORKFLOW                     │
└──────────────────────────────────────────────┘

Admin Login
    ↓
Authentication against User model (is_staff=True)
    ↓
Admin Dashboard
    ├─ Statistics:
    │  ├─ Total Applications
    │  ├─ Pending Count
    │  ├─ Approved Count
    │  └─ Rejected Count
    │
    ├─ Navigation:
    │  ├─ View All Loans
    │  ├─ View All Users
    │  ├─ View All Predictions
    │  └─ Manage Settings
    │
    └─ Management Options:
       ├─ Filter loans by status
       ├─ View loan details
       ├─ Update status/remarks
       ├─ View user profiles
       ├─ Delete users (cascade)
       └─ Configure prediction threshold
```

---

## Data Flow & Transformation

### Input Data Schema (User Submission)

| Field | Type | Range | Example |
|-------|------|-------|---------|
| Gender | Categorical | {Male, Female} | Male |
| Married | Categorical | {Yes, No} | Yes |
| Dependents | Categorical | {0, 1, 2, 3+} | 2 |
| Education | Categorical | {Graduate, Not Graduate} | Graduate |
| Self_Employed | Categorical | {Yes, No} | No |
| ApplicantIncome | Numerical | 0-∞ | 5000 |
| CoapplicantIncome | Numerical | 0-∞ | 2000 |
| LoanAmount | Numerical | 0-∞ | 120 |
| Loan_Amount_Term | Numerical | 0-∞ | 360 |
| Credit_History | Categorical | {0, 1} | 1 |
| Property_Area | Categorical | {Urban, Semiurban, Rural} | Urban |

### Data Preprocessing Pipeline

```
Raw Input (Dict)
    ↓
┌──────────────────────────────────┐
│ Step 1: DataFrame Creation       │
│ Convert dict → Pandas DataFrame  │
│ Shape: (1, 11)                   │
└──────────┬───────────────────────┘
           ↓
┌──────────────────────────────────────────────────┐
│ Step 2: Categorical Feature Encoding             │
│                                                  │
│ OneHotEncoder applied to:                        │
│ - Gender        → Gender_Male, Gender_Female    │
│ - Married       → Married_Yes, Married_No       │
│ - Dependents    → Dep_0, Dep_1, Dep_2, Dep_3+  │
│ - Education     → Edu_Graduate, Edu_NotGrad    │
│ - Self_Employed → SelfEmp_Yes, SelfEmp_No      │
│ - Credit_History→ Credit_0, Credit_1            │
│ - Property_Area → Area_Urban, Area_Semi, Rural  │
│                                                  │
│ Output: ~30+ binary features (sparse matrix)   │
└──────────┬───────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────┐
│ Step 3: Numerical Feature Scaling                │
│                                                  │
│ StandardScaler applied to:                       │
│ - ApplicantIncome      → mean=μ, std=σ          │
│ - CoapplicantIncome    → mean=μ, std=σ          │
│ - LoanAmount           → mean=μ, std=σ          │
│ - Loan_Amount_Term     → mean=μ, std=σ          │
│                                                  │
│ Formula: X_scaled = (X - μ) / σ                 │
│                                                  │
│ Standardized to N(0,1) distribution             │
└──────────┬───────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────┐
│ Step 4: Handle Missing Values                    │
│                                                  │
│ SimpleImputer applied:                           │
│ - Strategy: mean (for numerical)                │
│ - Strategy: most_frequent (for categorical)    │
│                                                  │
│ But form requires all fields, so rare            │
└──────────┬───────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────┐
│ Step 5: Final Feature Matrix                     │
│                                                  │
│ Shape: (1, F) where F = total features          │
│ Values: Mix of {0,1} (one-hot) and scaled vals  │
│                                                  │
│ Ready for XGBoost model                         │
└──────────┬───────────────────────────────────────┘
           ↓
      [To XGBoost Model]
```

### Output Transformation

```
XGBoost Model
    ↓
Raw Logit Score (tree ensemble output)
    ↓
Apply Sigmoid Function
    └─ Score → [0.0, 1.0] probability
    ↓
Probability of Default Class
    ├─ If prob > 0.65 → Pred=1 (Default)
    └─ If prob ≤ 0.65 → Pred=0 (Non-Default)
    ↓
SHAP Explainability
    ├─ Compute SHAP values for each feature
    ├─ Sort by absolute contribution
    └─ Return top 5 reasons
    ↓
Final Output Dict
{
    "prediction": 1,
    "probability": 0.73,
    "top_reasons": [
        ("Credit_History", 0.22),
        ("ApplicantIncome", 0.15),
        ("LoanAmount", -0.08),
        ...
    ]
}
```

---

## Feature Engineering

### Categorical Features (5)

| Feature | Values | Encoding | Purpose |
|---------|--------|----------|---------|
| **Gender** | Male, Female | One-Hot | Demographic info |
| **Married** | Yes, No | One-Hot | Marital status proxy for stability |
| **Dependents** | 0,1,2,3+ | One-Hot | Financial obligations |
| **Education** | Graduate, NotGraduate | One-Hot | Income/employment potential |
| **Self_Employed** | Yes, No | One-Hot | Income stability indicator |

### Numerical Features (6)

| Feature | Unit | Significance | Scaling |
|---------|------|--------------|---------|
| **ApplicantIncome** | Monthly | Primary income source | StandardScaler |
| **CoapplicantIncome** | Monthly | Joint income availability | StandardScaler |
| **LoanAmount** | Thousands | Debt-to-income ratio | StandardScaler |
| **Loan_Amount_Term** | Months | Repayment timeline | StandardScaler |
| **Credit_History** | Binary | Past payment behavior | N/A (already binary) |
| **Property_Area** | Categorical | Geographic risk | One-Hot |

### Feature Importance (from SHAP)

SHAP values indicate which features have most impact on predictions:
- **Positive SHAP:** Feature increases default probability
- **Negative SHAP:** Feature decreases default probability
- **Magnitude:** How much feature impacts prediction

Example interpretation:
```
Credit_History = +0.22  → No credit history strongly increases default risk
ApplicantIncome = -0.15 → Higher income reduces default risk
LoanAmount = +0.08      → Higher requested loan slightly increases risk
```

---

## Model Inference Pipeline

### Loading Model (First-Time Cache)

```python
# predictions/predictor.py

def get_model():
    """
    Lazy loading + caching of ML model
    First call: Load from disk (slow, ~500ms)
    Subsequent calls: Return from memory (fast, <1ms)
    """
    global MODEL
    if MODEL is None:
        logger.info("Loading model from disk...")
        MODEL = joblib.load(MODEL_PATH)  # /ml/models/pipeline_xgb.joblib
        logger.info("Model loaded successfully")
    return MODEL

# Same pattern for SHAP explainer and feature names
```

### Prediction Steps

```python
def predict_with_explanations(input_dict):
    """
    Complete prediction pipeline
    """
    
    # 1. Load cached components
    model = get_model()                 # XGBoost pipeline
    explainer = get_explainer()         # SHAP explainer
    feature_names = get_feature_names() # Column names
    
    # 2. Convert to DataFrame
    df = pd.DataFrame([input_dict])  # (1, 11)
    
    # 3. Prediction
    pred_class = model.predict(df)[0]              # 0 or 1
    pred_proba = model.predict_proba(df)[0][1]   # P(default)
    
    # 4. Feature transformation (for SHAP)
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(df)
    
    # 5. SHAP values
    shap_values = explainer.shap_values(X_transformed)[0]
    
    # 6. Map to feature names & sort
    shap_dict = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # 7. Return top 5
    return {
        "prediction": int(pred_class),
        "probability": float(pred_proba),
        "top_reasons": shap_dict[:5]
    }
```

### Caching Strategy

```
Request 1 (t=0ms):
  ├─ Load model from disk: 500ms
  ├─ Predict: 50ms
  └─ Total: 550ms ✓ (acceptable for first request)

Request 2 (t=600ms):
  ├─ Use cached model: 0ms (already in RAM)
  ├─ Predict: 50ms
  └─ Total: 50ms ✓ (fast!)

Request 3-1000 (t=650ms+):
  ├─ Use cached model: 0ms
  ├─ Predict: 50ms each
  └─ Total: 50ms ✓ (consistent)
```

**Benefits:**
- First prediction slower but acceptable
- Subsequent predictions very fast (50ms)
- No disk I/O after first load
- Model stays in memory as long as Django process runs

---

## Explainability (SHAP)

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theoretic approach that explains predictions by calculating each feature's contribution to moving the prediction from base value to final prediction.

### SHAP Value Interpretation

```
Base Value (Model Background) = E[Prediction] ≈ 0.5

For Applicant with:
├─ Income: $5,000 (below avg)
├─ Loan: $120,000 (high)
└─ Credit: Good (positive)

SHAP calculation:
┌────────────────────────────────────────┐
│ Base Value:                    0.50    │
│ + Credit_History SHAP:        +0.22    │
│ + ApplicantIncome SHAP:       -0.15    │
│ + LoanAmount SHAP:            +0.10    │
│ + Other features:             +0.06    │
├────────────────────────────────────────┤
│ Final Prediction:              0.73    │ → DEFAULT RISK (73%)
└────────────────────────────────────────┘

Top Reasons for HIGH default risk:
1. No credit history (lacks repayment proof)
2. High loan amount (large obligation)
3. Lower income (repayment capacity concern)
```

### Generating SHAP Values

```python
# In predictor.py
explainer = joblib.load(EXPLAINER_PATH)

# X_transformed: preprocessed features (one-hot encoded, scaled)
shap_values = explainer.shap_values(X_transformed)

# Result: array of shape (1, n_features)
# Each value: contribution of feature to prediction
```

### Displaying SHAP Results to User

In result.html template:
```
Prediction: LIKELY DEFAULT (73% probability)

Top Factors:
1. Credit History (Impact: +0.22) - No established credit history
2. Loan Amount (Impact: +0.10) - Requesting relatively high loan
3. Income Level (Impact: -0.15) - Income supports repayment
4. Employment Type (Impact: -0.08) - Stable employment
5. Property Area (Impact: +0.05) - Geographic considerations
```

---

## System Components

### 1. Frontend (React + Vite)

**Structure:**
```
frontend/
├── src/
│   ├── App.jsx                    # Root component
│   ├── components/
│   │   ├── LoanPredictionForm.jsx # Main form
│   │   ├── LoanField.jsx          # Individual field
│   │   └── loanFormFields.js      # Field config
│   ├── App.css                    # Global styles
│   └── main.jsx                   # Entry point
├── vite.config.js                 # Build config
└── package.json                   # Dependencies
```

**Components:**
- `LoanPredictionForm`: Parent form wrapper
- `LoanField`: Reusable field component
- `loanFormFields`: Configuration array

**Build Process:**
```bash
npm install      # Install dependencies
npm run build    # Build with Vite
# Output: predictions/static/predictions/react/
```

### 2. Backend (Django)

**Django Apps:**

**predictions/**
- User registration & login
- Loan application form handling
- ML prediction orchestration
- Prediction history display
- User profile management

**admin_panel/**
- Admin authentication
- Loan review dashboard
- User management
- Prediction analytics

**loan_predictor/**
- Django settings
- URL routing
- Middleware configuration

### 3. ML Pipeline (predictions/predictor.py)

```python
# Responsibilities:
├─ Model loading/caching
├─ Feature preprocessing
├─ XGBoost inference
├─ SHAP explainability
└─ Result formatting
```

### 4. Database (SQLite)

**Key Tables:**

**auth_user** (Django built-in)
```sql
├─ id: Integer (PK)
├─ username: String (unique)
├─ password: String (hashed)
├─ email: String
├─ first_name, last_name: String
├─ is_staff: Boolean (admin flag)
└─ date_joined: DateTime
```

**predictions_userprofile**
```sql
├─ id: Integer (PK)
├─ user_id: Integer (FK to auth_user)
├─ mobile_number: String
└─ address: TextField
```

**predictions_loanapplication**
```sql
├─ id: Integer (PK)
├─ user_id: Integer (FK)
├─ gender, married, dependents: String
├─ education, self_employed: String
├─ applicant_income, coapplicant_income: Float
├─ loan_amount, loan_amount_term: Float
├─ credit_history: Integer
├─ property_area: String
└─ created_at: DateTime
```

**predictions_predictionresult**
```sql
├─ id: Integer (PK)
├─ application_id: Integer (FK, unique)
├─ predicted_default: Boolean
├─ probability: Float (0.0-1.0)
├─ model_version: String
└─ created_at: DateTime
```

**predictions_loanapplicationformdata**
```sql
├─ id: Integer (PK)
├─ user_id: Integer (FK)
├─ prediction_id: Integer (FK)
├─ loan_amount: Float
├─ loan_tenure: Integer
├─ loan_purpose: String
├─ status: Choice (Pending/Approved/Rejected)
├─ remark: TextField
└─ applied_at: DateTime
```

---

## Database Schema

### Entity Relationship Diagram

```
┌──────────────────┐
│   auth_user      │
├──────────────────┤
│ id (PK)          │
│ username (U)     │◄────────────────┐
│ password         │                 │
│ email            │                 │
│ is_staff         │                 │
└──────────────────┘                 │
        ▲                             │
        │                             │
        │ 1:1                         │ 1:1
        │                             │
┌───────┴──────────────┐    ┌─────────┴─────────────┐
│ predictions_         │    │ predictions_          │
│ userprofile          │    │ loanapplication       │
├──────────────────────┤    ├───────────────────────┤
│ id (PK)              │    │ id (PK)               │
│ user_id (FK, U)      │    │ user_id (FK)          │
│ mobile_number        │    │ gender                │
│ address              │    │ married               │
└──────────────────────┘    │ ... (9 more fields)   │
                            │ created_at            │
                            └───────────┬───────────┘
                                        │ 1:1
                                        │
                            ┌───────────▼───────────┐
                            │ predictions_          │
                            │ predictionresult      │
                            ├───────────────────────┤
                            │ id (PK)               │
                            │ application_id (FK,U)│
                            │ predicted_default     │
                            │ probability           │
                            │ model_version         │
                            │ created_at            │
                            └───────────┬───────────┘
                                        │ 1:N
                                        │
                            ┌───────────▼────────────────┐
                            │ predictions_               │
                            │ loanapplicationformdata    │
                            ├────────────────────────────┤
                            │ id (PK)                    │
                            │ prediction_id (FK)         │
                            │ user_id (FK)               │
                            │ loan_amount                │
                            │ loan_tenure                │
                            │ status (Pending/...)       │
                            │ remark                     │
                            │ applied_at                 │
                            └────────────────────────────┘
```

**Key Relationships:**
- User → UserProfile (1:1)
- User → LoanApplication (1:N) - multiple predictions per user
- LoanApplication → PredictionResult (1:1) - one prediction per app
- PredictionResult → LoanApplicationFormData (1:N) - can have formal apps

---

## API Integration Points

### 1. User Loan Prediction API

**Endpoint:** `/predict/` (POST)

**Authentication:** Required (login_required decorator)

**Request:**
```http
POST /predict/ HTTP/1.1
Content-Type: application/x-www-form-urlencoded

gender=Male&married=Yes&dependents=1&education=Graduate&
self_employed=No&applicant_income=5000&coapplicant_income=2000&
loan_amount=120&loan_amount_term=360&credit_history=1&
property_area=Urban&csrfmiddlewaretoken=xxxxx
```

**Backend Processing:**
```python
@login_required
def predict_view(request):
    form = LoanApplicationForm(request.POST)
    if form.is_valid():
        # 1. Save LoanApplication
        app = form.save(commit=False)
        app.user = request.user
        app.save()
        
        # 2. Extract features
        input_data = {...}
        
        # 3. Get threshold
        threshold = PredictionConfig.objects.first().threshold
        
        # 4. Get predictions
        model = get_model()
        proba = model.predict_proba(pd.DataFrame([input_data]))[:, 1][0]
        
        # 5. Apply threshold
        predicted = bool(proba > threshold)
        
        # 6. Save PredictionResult
        pred = PredictionResult.objects.create(
            application=app,
            predicted_default=predicted,
            probability=proba
        )
        
        # 7. Return result
        return render(request, 'predictions/result.html', {
            'prediction': pred,
            'application': app
        })
```

**Response (HTML):**
```html
<!-- result.html rendered with context -->
<div class="result">
    <h2>Prediction Result</h2>
    <p>Status: LIKELY DEFAULT (73%)</p>
    <div class="top-reasons">
        <!-- SHAP explanations displayed -->
    </div>
    <a href="{% url 'apply_loan' %}">Apply for Formal Loan</a>
</div>
```

### 2. Admin Loan Review API

**Endpoint:** `/admin-panel/loans/<id>/` (GET, POST)

**Authentication:** Admin only (@user_passes_test)

**GET:** Display loan details

**POST:** Update status and remarks
```python
def loan_detail_view(request, id):
    loan = get_object_or_404(LoanApplicationFormData, id=id)
    
    if request.method == 'POST':
        loan.status = request.POST.get('status')  # Approve/Reject
        loan.remark = request.POST.get('remark')
        loan.save()
        return redirect('admin_loan_list')
    
    return render(request, 'loan_detail.html', {'loan': loan})
```

### 3. Model Configuration API

**Endpoint:** `/admin-panel/settings/` (conceptual - not yet implemented)

**Future Enhancement:** Allow admins to adjust prediction threshold via UI

```python
# In PredictionConfig model
class PredictionConfig(models.Model):
    threshold = models.FloatField(default=0.65)
    
    # Could be extended to:
    # - model_version selection
    # - feature importance weights
    # - dynamic threshold by user segment
```

---

## Security & Authentication

### Authentication Flow

```
1. User Registration
   └─ Password hashed with Django's pbkdf2
   └─ Stored in auth_user table
   └─ UserProfile created separately

2. User Login
   ├─ Username + password submitted
   ├─ Django authenticates against auth_user
   ├─ Session cookie created (secure, httponly)
   └─ User is logged in

3. Subsequent Requests
   ├─ Session cookie attached to request
   ├─ Django middleware verifies session
   ├─ request.user populated
   └─ Access granted to protected views
```

### Authorization

**User Views Protection:**
```python
@login_required(login_url='login')
def predict_view(request):
    # Only authenticated users
    app.user = request.user  # Current user enforced
```

**Admin Views Protection:**
```python
@user_passes_test(admin_required, login_url='admin_login')
def dashboard_view(request):
    # Only staff users (is_staff=True)
    # Checked on every request
```

### CSRF Protection

All forms include Django CSRF token:
```html
<form method="post" action="/predict/">
    {% csrf_token %}
    <!-- Form fields -->
</form>
```

Backend validates token before processing:
```python
# CsrfViewMiddleware ensures POST requests have valid token
```

### Data Security

**Sensitive Data Handling:**
- Passwords: Hashed with PBKDF2
- Sessions: Secure cookies (httponly, secure flags)
- Database: SQLite (OK for dev, PostgreSQL recommended for prod)
- API: Only Django views exposed (no direct ML model access)

---

## Summary

This comprehensive documentation covers:

✅ **System Architecture** - 3-layer architecture with frontend, backend, ML
✅ **ML Model** - XGBoost with 300 trees, 11 features, SHAP explainability
✅ **Complete Workflows** - Registration, prediction, formal application, admin review
✅ **Data Flow** - Input → Preprocessing → Prediction → Explanation
✅ **Feature Engineering** - 5 categorical + 6 numerical features
✅ **Inference Pipeline** - Model loading, caching, prediction steps
✅ **SHAP Explainability** - How predictions are explained to users
✅ **System Components** - React frontend, Django backend, ML pipeline
✅ **Database Schema** - 5 key tables with relationships
✅ **API Integration** - Loan prediction, admin review, configuration
✅ **Security** - Authentication, authorization, CSRF, password hashing

---

## Next Steps & Enhancements

**Immediate:**
- [ ] Add API documentation (Swagger/DRF)
- [ ] Implement comprehensive test suite
- [ ] Set up CI/CD pipeline

**Short-term:**
- [ ] Docker containerization
- [ ] Migrate to PostgreSQL
- [ ] Add model versioning/rollback
- [ ] Implement A/B testing for threshold

**Long-term:**
- [ ] Real-time model monitoring
- [ ] Automated model retraining
- [ ] Multi-model ensemble
- [ ] Explainable AI dashboard for regulators
- [ ] Fairness/bias audit tools

---

**Document Version:** 1.0
**Last Updated:** May 2026
**Project:** Loan Predictor ML System
