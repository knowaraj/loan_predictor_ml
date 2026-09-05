# Comprehensive Data Pipeline Analysis
## Loan Predictor ML System

---

## Table of Contents
1. [Raw Data Overview](#raw-data-overview)
2. [Data Exploration & Analysis](#data-exploration--analysis)
3. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Data Splitting & Validation](#data-splitting--validation)
6. [Model Training Details](#model-training-details)
7. [Training Parameters & Configuration](#training-parameters--configuration)
8. [Model Evaluation & Metrics](#model-evaluation--metrics)
9. [SHAP Explainability](#shap-explainability)
10. [Complete Training Script Summary](#complete-training-script-summary)

---

## Raw Data Overview

### Data Files
- **Primary Dataset:** `ml/data/loan_data.csv` (used by train2.py and exp.ipynb)
- **Secondary Dataset:** `ml/data/loan_data2.csv` (used by train.py for XGBoost-like model)

### Dataset Characteristics
- **Target Variable:** `Loan_Status` (Categorical: 'Y' = Non-default, 'N' = Default)
- **Total Features:** 11 features
  - **Categorical (5):** Gender, Married, Dependents, Education, Self_Employed, Property_Area
  - **Numerical (6):** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

### Target Variable Mapping
```python
# In training scripts:
y = df[TARGET].map({'Y': 0, 'N': 1})  # 1 = Defaulter/Rejection, 0 = Non-defaulter/Approval
```

---

## Data Exploration & Analysis

### Exploratory Data Analysis (EDA)
**File:** `ml/exp.ipynb`

#### EDA Steps Performed:
1. **Data Loading**
   ```python
   df = pd.read_csv(r'path/to/loan_data.csv')
   ```

2. **Initial Data Inspection**
   - `df.head()` - First few records examination
   - `df.info()` - Column types and non-null counts
   - `df.isna().sum()` - Missing value analysis

3. **Feature Definition**
   ```python
   cat_features = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
   num_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
   ```

4. **Exploratory Models Tested**
   - **RandomForestClassifier:** n_estimators=200, class_weight='balanced'
   - **XGBClassifier:** Binary logistic objective with GridSearchCV
   - **MLPClassifier (Neural Network):** Deep learning approach with hyperparameter tuning

5. **Hyperparameter Tuning (GridSearchCV)**
   - Tested multiple configurations for each model
   - Used 5-fold cross-validation
   - Scoring metrics: accuracy, ROC-AUC, F1-score

### ROC & Confusion Matrix Analysis
**File:** `ml/roc_confusion_matrix_analysis.ipynb`

#### Analysis Outputs:
1. **Confusion Matrices** - Training vs Test set comparison
2. **ROC Curves** - Model discrimination ability visualization
3. **Performance Metrics:**
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - F1-Score
   - ROC-AUC Score

4. **Additional Visualizations:**
   - Metrics comparison bar charts
   - ROC-AUC score comparison
   - Prediction distribution (test set)
   - Predicted probability distributions with decision threshold

---

## Data Preprocessing Pipeline

### Pipeline Architecture
The preprocessing uses **scikit-learn ColumnTransformer** with two parallel branches:

```
Raw Input DataFrame
    │
    ├─ Numerical Branch (5 features)
    │  ├─ SimpleImputer (strategy='median') → Handle missing values
    │  └─ StandardScaler → Normalize to N(0,1)
    │
    └─ Categorical Branch (6 features)
       ├─ SimpleImputer (strategy='constant', fill_value='Missing')
       └─ OneHotEncoder (handle_unknown='ignore', sparse_output=False)
    
    Combined Output: Processed Feature Matrix
```

### Preprocessing Implementation

#### Code from `train.py` and `exp.ipynb`:
```python
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])
```

### Preprocessing Steps Detailed

#### 1. Numerical Features Processing
- **Features:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- **Imputation:** Missing values replaced with **median** (robust to outliers)
- **Scaling:** **StandardScaler** - transforms to mean=0, std=1
  - Formula: `X_scaled = (X - mean) / std`
  - Reason: XGBoost is less sensitive to scaling, but StandardScaler still improves performance

#### 2. Categorical Features Processing
- **Features:** Gender, Married, Dependents, Education, Self_Employed, Property_Area
- **Imputation:** Missing values replaced with 'Missing' (constant strategy)
- **Encoding:** **OneHotEncoder**
  - Creates binary dummy variables for each category
  - `handle_unknown='ignore'` - handles unseen categories gracefully
  - Output: Sparse or dense matrix (configurable)

#### 3. Manual Data Cleaning (train.py specific)
```python
# Direct mapping (preprocessing before pipeline)
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df['Married'] = df['Married'].map({'Yes':1,'No':0})
df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0})
df['Property_Area'] = df['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})
df['Dependents'] = df['Dependents'].replace('3+', '3').astype(float)
df[TARGET] = df[TARGET].map({'Y':0,'N':1})

# Pre-fill missing values
for c in num_features:
    df[c] = df[c].fillna(df[c].median())
for c in cat_features:
    df[c] = df[c].astype(str).fillna('Missing')
```

### Missing Value Handling Strategy
- **Numerical:** Median imputation (resistant to outliers)
- **Categorical:** Most frequent value or 'Missing' placeholder
- **Rationale:** Median is robust for skewed income distributions

---

## Feature Engineering

### Categorical Features (6 total)

| Feature | Values | Encoding Method | Purpose |
|---------|--------|-----------------|---------|
| **Gender** | Male, Female | One-Hot / Label | Demographic factor |
| **Married** | Yes, No | One-Hot / Label | Marital stability indicator |
| **Dependents** | 0, 1, 2, 3+ | One-Hot / Numeric | Financial obligations |
| **Education** | Graduate, Not Graduate | One-Hot / Label | Income/employment proxy |
| **Self_Employed** | Yes, No | One-Hot / Label | Income stability indicator |
| **Property_Area** | Rural, Semiurban, Urban | One-Hot / Label | Geographic risk factor |

### Numerical Features (6 total)

| Feature | Unit | Range | Significance | Transformation |
|---------|------|-------|--------------|-----------------|
| **ApplicantIncome** | Monthly (₹) | 0-∞ | Primary income | StandardScaler |
| **CoapplicantIncome** | Monthly (₹) | 0-∞ | Joint income | StandardScaler |
| **LoanAmount** | Thousands (₹) | 0-∞ | Loan request size | StandardScaler |
| **Loan_Amount_Term** | Months | 0-∞ | Repayment period | StandardScaler |
| **Credit_History** | Binary (0/1) | {0, 1} | Past payment behavior | No scaling (binary) |

### Feature Importance (from SHAP Analysis)
- **Credit_History:** Strong positive impact on default prediction
- **ApplicantIncome:** Negative impact (higher income → lower default)
- **LoanAmount:** Moderate positive impact on default risk
- **Property_Area:** Geographic risk variations
- **Other features:** Supporting factors in prediction

### Total Features After Preprocessing
- **One-Hot Encoded Categorical:** ~25-30 binary features
- **Scaled Numerical:** 5-6 normalized features
- **Total Input Features to Model:** ~35+ features

---

## Data Splitting & Validation

### Train-Test Split Configuration

#### Primary Split (train.py & train2.py)
```python
X_temp, X_test_df, y_temp, y_test = train_test_split(
    X_df, y, 
    test_size=0.2,           # 80-20 split
    stratify=y,              # Preserves class distribution
    random_state=42          # Reproducibility
)
```
- **Training Data:** 80% of dataset
- **Test Data:** 20% of dataset
- **Stratification:** Maintains class balance in both sets

#### Validation Split for Early Stopping (train.py)
```python
if VALIDATION_FOR_EARLY_STOP > 0:
    val_frac = VALIDATION_FOR_EARLY_STOP / (1 - TEST_SIZE)
    # VALIDATION_FOR_EARLY_STOP = 0.1 (10% of training)
    
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_frac,    # 10% of 80% = 8% of total
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
```
- **Final Train:** ~72% of total dataset
- **Validation:** ~8% of total dataset (for early stopping)
- **Test:** ~20% of total dataset

### Validation Approach
- **3-way split:** Training, Validation (early stopping), Test
- **Early Stopping:** Monitors validation loss; stops if no improvement for N rounds
- **Class Stratification:** Ensures balanced class distribution in all splits

---

## Model Training Details

### Models Implemented

#### 1. **Random Forest (train2.py)**
```python
model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',    # Handles class imbalance
        random_state=42
    ))
])
```
- **Base Model:** Ensemble of 200 decision trees
- **Class Weight:** 'balanced' - auto-adjusts weights inversely proportional to class frequency
- **Advantage:** Fast training, good baseline performance

#### 2. **XGBoost Classifier (exp.ipynb)**
```python
model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    ))
])
```
- **Objective:** Binary logistic regression
- **Evaluation Metric:** Log loss (cross-entropy)
- **Note:** Scikit-learn wrapper around XGBoost

#### 3. **Custom XGBoost-Like Model (train.py - FINAL MODEL)**
A **custom-built gradient boosting implementation** with:

**Architecture:**
```
RefinedXGBLike Model
├─ Base Score Initialization (log-odds)
├─ Iterative Tree Building (300 rounds)
├─ Gradient & Hessian Computation
├─ Row/Column Sampling (subsampling & feature subsampling)
├─ Tree-based Splitting (gain-based)
├─ Leaf Weight Optimization (closed-form XGBoost formula)
└─ Learning Rate Scaling (shrinkage)
```

**Key Components:**
- **XGBStyleTree:** Custom tree implementation using gradient boosting
- **Sigmoid Activation:** Converts raw scores to probabilities
- **Early Stopping:** Based on validation loss

#### 4. **Neural Network (exp.ipynb - MLP)**
```python
model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', MLPClassifier(
        max_iter=500,
        random_state=42,
        early_stopping=True
    ))
])
```
- **Architecture:** Variable hidden layer sizes
- **Activation:** ReLU/Tanh
- **Early Stopping:** Prevents overfitting

---

## Training Parameters & Configuration

### Final Production Model Parameters (train.py)

```python
# Model Hyperparameters
N_ESTIMATORS = 300              # Number of boosting rounds
LEARNING_RATE = 0.05            # Shrinkage (eta) - reduces overfit
MAX_DEPTH = 5                   # Maximum tree depth - shallow trees
MIN_SAMPLES_LEAF = 8            # Minimum samples in leaf node
LAMBDA = 1.0                    # L2 regularization parameter
SUBSAMPLE = 0.8                 # Row sampling - 80% of samples per tree
COLSAMPLE_BYTREE = 0.8          # Column sampling - 80% of features per tree
EARLY_STOPPING_ROUNDS = 20      # Stop if val loss doesn't improve for 20 rounds
GAMMA = 0.0                     # Minimum loss reduction for split
MIN_CHILD_WEIGHT = 1.0          # Minimum sum of instance weights in leaf
```

### Data Split Configuration (train.py)
```python
TEST_SIZE = 0.2                 # 20% test set
RANDOM_STATE = 42               # For reproducibility
VALIDATION_FOR_EARLY_STOP = 0.1 # 10% validation set for early stopping
```

### Regularization Strategy
- **L2 Regularization (Lambda=1.0):** Prevents weight explosion
- **Subsampling (0.8):** Reduces overfitting via stochastic gradient descent
- **Column Sampling (0.8):** Feature-level stochasticity
- **Learning Rate (0.05):** Small shrinkage for stable optimization
- **Max Depth (5):** Shallow trees prevent overfitting
- **Early Stopping (20 rounds):** Stops training when validation loss plateaus

---

## Model Evaluation & Metrics

### Evaluation Metrics Computed

#### Training Set Metrics
- **Accuracy:** Overall correctness of predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall (Sensitivity):** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the ROC curve (0.5=random, 1.0=perfect)

#### Test Set Metrics
Same metrics as training set for generalization assessment

### Classification Report Output
```python
print(classification_report(y_test, y_test_pred))
# Outputs: Precision, Recall, F1-Score per class
```

### Confusion Matrix Analysis
```
                   Predicted
                Negative  Positive
Actual Negative    TN        FP
       Positive    FN        TP
```

### Model Performance Interpretation (from notebooks)
- **Train vs Test Gap:** Indicates overfitting/underfitting
- **ROC-AUC Close to 1.0:** Good discrimination ability
- **Balanced Precision/Recall:** Good threshold setting
- **High F1-Score:** Model handles both FP and FN well

### Evaluation Visualizations Generated
1. **Confusion Matrices:** Side-by-side training vs test
2. **ROC Curves:** Train and test set comparison
3. **Performance Metrics Bar Charts:** Accuracy, Precision, Recall, F1
4. **ROC-AUC Comparison:** Training vs Test
5. **Prediction Distribution:** Class balance in predictions
6. **Probability Distribution:** Histogram of predicted probabilities with threshold line

---

## SHAP Explainability

### SHAP Integration (predictor.py)

#### SHAP Explainer Loading
```python
def get_explainer() -> Any:
    """Load cached SHAP explainer for feature importance."""
    global EXPLAINER
    if EXPLAINER is None:
        EXPLAINER = joblib.load(EXPLAINER_PATH)
    return EXPLAINER
```

#### SHAP Values Computation
```python
# In predict_with_explanations():
preprocessor = model.named_steps["preprocessor"]
transformed = preprocessor.transform(df)
shap_values = explainer.shap_values(transformed)[0]
```

#### Feature Importance Extraction
```python
# Map SHAP values to feature names and sort
shap_dict = sorted(
    zip(feature_names, shap_values),
    key=lambda x: abs(x[1]),      # Sort by absolute impact
    reverse=True
)
top_reasons = shap_dict[:5]  # Top 5 most influential features
```

### SHAP Interpretation
- **Positive SHAP Value:** Feature increases default probability
- **Negative SHAP Value:** Feature decreases default probability
- **Magnitude (|SHAP|):** Strength of feature's influence on prediction
- **Top 5 Reasons:** Most influential features for that specific prediction

#### Prediction Output with SHAP
```python
{
    "prediction": 1,           # 1=Default, 0=Non-default
    "probability": 0.73,       # 73% probability of default
    "top_reasons": [           # Top 5 SHAP-based features
        ("Credit_History", 0.22),      # Strongest positive impact
        ("ApplicantIncome", -0.15),    # Negative (protective) impact
        ("LoanAmount", 0.08),
        ("Married", 0.05),
        ("Property_Area", -0.03)
    ]
}
```

### SHAP Files Saved
- **shap_explainer.joblib:** Pre-computed SHAP explainer object (cached)
- **feature_names.joblib:** List of all feature names post-preprocessing

---

## Complete Training Script Summary

### Script Flow Diagram (train.py)

```
START
  │
  ├─ Load CSV (loan_data2.csv)
  │
  ├─ Data Cleaning:
  │  ├─ Map categorical values (Gender, Married, Education, etc.)
  │  ├─ Handle '3+' dependents → '3'
  │  ├─ Map target: {'Y':0, 'N':1}
  │  ├─ Fill numeric NaN → median
  │  └─ Fill categorical NaN → 'Missing'
  │
  ├─ Create Preprocessing Pipeline:
  │  ├─ Numerical: SimpleImputer(median) + StandardScaler
  │  └─ Categorical: SimpleImputer(constant) + OneHotEncoder
  │
  ├─ Train-Test Split (80-20, stratified)
  │  └─ Further split into Train (72%) / Validation (8%) / Test (20%)
  │
  ├─ Fit Preprocessor:
  │  └─ preprocessor.fit(X_train)
  │
  ├─ Transform All Sets:
  │  ├─ X_train_transformed = preprocessor.transform(X_train)
  │  ├─ X_val_transformed = preprocessor.transform(X_val)
  │  └─ X_test_transformed = preprocessor.transform(X_test)
  │
  ├─ Initialize RefinedXGBLike Model:
  │  ├─ n_estimators=300
  │  ├─ learning_rate=0.05
  │  ├─ max_depth=5
  │  ├─ subsample=0.8
  │  ├─ colsample_bytree=0.8
  │  └─ early_stopping_rounds=20
  │
  ├─ Train Model:
  │  └─ model.fit(X_train_transformed, y_train, 
  │              X_val=X_val_transformed, y_val=y_val)
  │  
  │  [During Training]:
  │  For each of 300 boosting rounds:
  │    1. Compute gradients & hessians from sigmoid(F)
  │    2. Sample rows (80%) and columns (80%)
  │    3. Build tree using gradient-boosting gain criterion
  │    4. Compute leaf weights using closed-form formula
  │    5. Update raw scores F += learning_rate * update
  │    6. Log training loss
  │    7. Compute validation loss
  │    8. Check early stopping criterion
  │
  ├─ Evaluate on Train Set:
  │  ├─ y_pred = model.predict(X_train_transformed)
  │  ├─ y_proba = model.predict_proba(X_train_transformed)
  │  ├─ Accuracy = accuracy_score(y_train, y_pred)
  │  ├─ ROC-AUC = roc_auc_score(y_train, y_proba)
  │  └─ Print classification_report
  │
  ├─ Evaluate on Test Set:
  │  ├─ y_pred = model.predict(X_test_transformed)
  │  ├─ y_proba = model.predict_proba(X_test_transformed)
  │  ├─ Accuracy = accuracy_score(y_test, y_pred)
  │  ├─ ROC-AUC = roc_auc_score(y_test, y_proba)
  │  └─ Print classification_report
  │
  ├─ Save Pipeline:
  │  └─ full_pipeline = {'preprocessor': preprocessor, 'model': model}
  │     joblib.dump(full_pipeline, 'ml/models/pipeline_xgb_like_fixed.joblib')
  │
  END
```

### Saved Artifacts

**Primary Model File:**
- `ml/models/pipeline_xgb_like_fixed.joblib` - Complete pipeline (preprocessor + model)

**Alternative Models:**
- `ml/models/pipeline.joblib` - RandomForest baseline (train2.py)
- `ml/models/pipeline_xgb.joblib` - Scikit-learn XGBoost wrapper
- `ml/models/pipeline_xgb_boost.joblib` - Alternative XGBoost configuration

**Supporting Files:**
- `ml/models/shap_explainer.joblib` - SHAP explainer (pre-computed)
- `ml/models/feature_names.joblib` - Feature names list (post-preprocessing)

---

## Production Inference Pipeline

### Prediction Execution (predictor.py)

```
User Input Dictionary
  │
  ├─ Convert to DataFrame
  │  └─ df = pd.DataFrame([input_dict])
  │
  ├─ Get Cached Model
  │  └─ model = get_model()
  │
  ├─ Run Prediction:
  │  ├─ pred = model.predict(df)[0]           # Class label (0 or 1)
  │  ├─ proba = model.predict_proba(df)[0][1] # Probability of class 1
  │  │
  │  └─ [Behind the scenes]:
  │     ├─ preprocessor.transform(df)
  │     │  ├─ One-hot encode categories
  │     │  ├─ StandardScale numericals
  │     │  └─ Output: transformed features
  │     │
  │     └─ model.predict_proba(transformed)
  │        ├─ Run through 300 trees
  │        ├─ Aggregate predictions (weighted sum)
  │        ├─ Apply sigmoid: P = 1 / (1 + e^(-raw_score))
  │        └─ Return probability
  │
  ├─ Apply Threshold (from PredictionConfig):
  │  └─ threshold = 0.65 (default, configurable)
  │     if proba > 0.65: pred = 1 (DEFAULT)
  │     else: pred = 0 (NON-DEFAULT)
  │
  ├─ Compute SHAP Values:
  │  ├─ explainer = get_explainer()
  │  ├─ shap_values = explainer.shap_values(transformed)
  │  ├─ Map to feature names
  │  └─ Sort by |shap_value|, take top 5
  │
  ├─ Save Prediction to Database:
  │  └─ PredictionResult.objects.create(
  │       application=loan_app,
  │       predicted_default=pred,
  │       probability=proba,
  │       model_version='v1'
  │     )
  │
  └─ Return Output Dict:
     {
         "prediction": int,
         "probability": float,
         "top_reasons": [(feature, shap_value), ...]
     }
```

---

## Key Insights & Summary

### Data Pipeline Characteristics
1. **Binary Classification Problem:** Default (1) vs Non-Default (0)
2. **Mixed Feature Types:** 6 categorical + 5 numerical
3. **Preprocessing:** StandardScaler for numerical, OneHotEncoder for categorical
4. **Imputation:** Median for numerical, mode/constant for categorical
5. **Class Imbalance Handling:** Stratified splits preserve class distribution

### Model Characteristics
1. **Ensemble Method:** Gradient boosting (300 trees, shallow depth)
2. **Regularization:** Heavy (subsample=0.8, colsample=0.8, lambda=1.0)
3. **Early Stopping:** Based on validation loss (20-round patience)
4. **Output:** Probability [0.0-1.0] with configurable threshold (0.65)
5. **Explainability:** SHAP provides top 5 feature explanations per prediction

### Training Approach
1. **Iterative Experimentation:** RandomForest → XGBoost → Custom XGBoost-like
2. **Hyperparameter Tuning:** GridSearchCV with cross-validation
3. **Validation Strategy:** 3-way split (train/val/test) with early stopping
4. **Evaluation:** Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

### Deployment Strategy
1. **Model Caching:** Models cached in memory to avoid repeated disk I/O
2. **SHAP Pre-computation:** Explainer pre-computed and stored
3. **Database Integration:** All predictions logged to SQLite for audit trail
4. **Threshold Configuration:** Configurable prediction threshold via admin panel

---

## References & File Locations

### Training Scripts
- [ml/train.py](../ml/train.py) - Custom XGBoost-like model (FINAL)
- [ml/train2.py](../ml/train2.py) - RandomForest baseline

### Exploratory Notebooks
- [ml/exp.ipynb](../ml/exp.ipynb) - EDA and model experimentation
- [ml/roc_confusion_matrix_analysis.ipynb](../ml/roc_confusion_matrix_analysis.ipynb) - Evaluation analysis

### Prediction Engine
- [predictions/predictor.py](../predictions/predictor.py) - Inference with SHAP

### Data Files
- [ml/data/loan_data.csv](../ml/data/loan_data.csv) - Primary dataset
- [ml/data/loan_data2.csv](../ml/data/loan_data2.csv) - Secondary dataset

### Model Artifacts
- [ml/models/pipeline_xgb_like_fixed.joblib](../ml/models/pipeline_xgb_like_fixed.joblib) - Production model
- [ml/models/shap_explainer.joblib](../ml/models/shap_explainer.joblib) - SHAP explainer
- [ml/models/feature_names.joblib](../ml/models/feature_names.joblib) - Feature names

---

**Analysis Generated:** May 10, 2026
**Analyst:** GitHub Copilot ML Pipeline Researcher
