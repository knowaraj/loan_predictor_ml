# train_fixed_xgb_like.py

import math
import pickle
from collections import defaultdict
import time
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import random

# --------------------------------------------
# CONFIG
# --------------------------------------------

CSV_PATH = r'C:\Users\LENOVO\Desktop\Learning\Project\loan_predictor\ml\data\loan_data2.csv'
TARGET = 'Loan_Status'

cat_features = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
num_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
FEATURES = cat_features + num_features

TEST_SIZE = 0.2
RANDOM_STATE = 42
VALIDATION_FOR_EARLY_STOP = 0.1

# Model hyperparameters
N_ESTIMATORS = 300
LEARNING_RATE = 0.05
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 8
LAMBDA = 1.0
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
EARLY_STOPPING_ROUNDS = 20
GAMMA = 0.0
MIN_CHILD_WEIGHT = 1.0

MODEL_OUT = "ml/models/pipeline_xgb_like_fixed.joblib"

# Ensure output directory exists
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)


# --------------------------------------------
# SIGMOID
# --------------------------------------------

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------
# HARD-CODED XGBOOST-STYLE TREE
# --------------------------------------------

class XGBStyleTree:
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None,
                     value=None, gain=None, G=None, H=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value    # leaf weight
            self.gain = gain
            self.G = G
            self.H = H

    def __init__(self, max_depth=3, min_samples_leaf=5,
                 lambda_reg=1.0, gamma=0.0, min_child_weight=1.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.root = None

    def _leaf_value(self, G, H):
        # closed-form leaf weight for XGBoost
        return -G / (H + self.lambda_reg)

    def fit(self, X, g, h):
        """Train tree using gradients g and h (second order). X, g, h are arrays of same length."""
        self.root = self._build_tree(X, g, h, depth=0)

    def _build_tree(self, X, g, h, depth):
        G = g.sum()
        H = h.sum()

        # Stopping conditions
        if depth >= self.max_depth or len(g) <= self.min_samples_leaf:
            return self.Node(value=self._leaf_value(G, H), G=G, H=H)

        best_gain = -1e12
        best_feature = None
        best_thresh = None
        best_left_idx = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            Xf = X[sorted_idx, feature]
            g_sorted = g[sorted_idx]
            h_sorted = h[sorted_idx]

            G_left = 0.0
            H_left = 0.0

            # Move split point from 1 to n-1
            for i in range(1, n_samples):
                G_left += g_sorted[i-1]
                H_left += h_sorted[i-1]

                G_right = G - G_left
                H_right = H - H_left

                # Enforce min child hessian (min_child_weight)
                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                # Skip if feature value doesn't change (no valid split)
                if Xf[i] == Xf[i-1]:
                    continue

                gain = (G_left**2) / (H_left + self.lambda_reg) + \
                       (G_right**2) / (H_right + self.lambda_reg) - \
                       (G**2) / (H + self.lambda_reg)

                gain -= self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = (Xf[i] + Xf[i-1]) / 2.0
                    # store indices of left side (in original indexing)
                    best_left_idx = sorted_idx[:i].copy()

        # If no good split -> make leaf
        if best_feature is None or best_gain <= 0:
            return self.Node(value=self._leaf_value(G, H), G=G, H=H)

        # Build left/right masks using best_left_idx
        left_mask = np.zeros(X.shape[0], dtype=bool)
        left_mask[best_left_idx] = True
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)

        return self.Node(
            feature=best_feature,
            threshold=best_thresh,
            left=left_node,
            right=right_node,
            gain=best_gain,
            G=G,
            H=H
        )

    def apply(self, X):
        """Return leaf id for each sample in X. Uses id(node) as leaf identifier."""
        leaf_ids = np.zeros(len(X), dtype=np.int64)
        for i, row in enumerate(X):
            node = self.root
            while node.value is None:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            leaf_ids[i] = id(node)
        return leaf_ids

    def predict(self, X):
        preds = np.zeros(len(X))
        for i, row in enumerate(X):
            node = self.root
            while node.value is None:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds[i] = node.value
        return preds


# --------------------------------------------
# REFYNED XGB-LIKE MODEL (uses XGBStyleTree)
# --------------------------------------------

class RefinedXGBLike:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=8, lambda_reg=1.0,
                 subsample=1.0, colsample_bytree=1.0,
                 random_state=None, early_stopping_rounds=0, verbose=True,
                 gamma=0.0, min_child_weight=1.0):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lambda_reg = lambda_reg
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.gamma = gamma
        self.min_child_weight = min_child_weight

        self.base_score = 0.0
        self.trees = []
        self.leaf_weights = []
        self.col_indices = []

    def _init_base_score(self, y):
        pos = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.base_score = math.log(pos / (1 - pos))

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.RandomState(self.random_state)

        n_samples, n_features = X.shape
        self._init_base_score(y)

        F = np.full(n_samples, self.base_score)
        F_val = (np.full(len(X_val), self.base_score) if X_val is not None else None)

        best_val_loss = float("inf")
        rounds_no_improve = 0

        for m in range(self.n_estimators):
            p = sigmoid(F)
            g = p - y                      # gradient
            h = p * (1 - p)                # hessian

            # Row sampling
            if 0 < self.subsample < 1:
                row_idx = rng.choice(n_samples, size=max(2, int(self.subsample * n_samples)), replace=False)
            else:
                row_idx = np.arange(n_samples)

            # Column sampling
            if 0 < self.colsample_bytree < 1:
                col_idx = rng.choice(n_features, size=max(1, int(self.colsample_bytree * n_features)), replace=False)
            else:
                col_idx = np.arange(n_features)

            # Build tree using gradients/hessians from sampled rows
            tree = XGBStyleTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                lambda_reg=self.lambda_reg,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight
            )
            # Note: pass gradients/hessians corresponding to the sampled rows
            tree.fit(X[row_idx][:, col_idx], g[row_idx], h[row_idx])

            # Apply tree to full training set (on selected columns) to get leaf ids
            leaves = tree.apply(X[:, col_idx])

            # Aggregate G and H per leaf across the entire training set (not only sample)
            leaf_G = defaultdict(float)
            leaf_H = defaultdict(float)
            for i, leaf in enumerate(leaves):
                leaf_G[leaf] += g[i]
                leaf_H[leaf] += h[i]

            # Compute leaf weights (same formula XGBoost uses)
            leaf_w = {leaf: -leaf_G[leaf] / (leaf_H[leaf] + self.lambda_reg) for leaf in leaf_G}
            update = np.array([leaf_w.get(leaf, 0.0) for leaf in leaves])

            # Update raw score
            F += self.learning_rate * update

            self.trees.append(tree)
            self.leaf_weights.append(leaf_w)
            self.col_indices.append(col_idx)

            # Compute training loss (logloss)
            prob = sigmoid(F)
            eps = 1e-15
            train_loss = -np.mean(y * np.log(np.clip(prob, eps, 1 - eps)) +
                                  (1 - y) * np.log(np.clip(1 - prob, eps, 1 - eps)))

            # Validation update & loss
            if X_val is not None:
                leaves_val = tree.apply(X_val[:, col_idx])
                update_val = np.array([leaf_w.get(l, 0.0) for l in leaves_val])
                F_val += self.learning_rate * update_val
                pv = sigmoid(F_val)
                val_loss = -np.mean(y_val * np.log(np.clip(pv, eps, 1 - eps)) +
                                    (1 - y_val) * np.log(np.clip(1 - pv, eps, 1 - eps)))
            else:
                val_loss = None

            if self.verbose and (m % 10 == 0 or m == self.n_estimators - 1):
                if val_loss is not None:
                    print(f"[{m+1}/{self.n_estimators}] train_logloss={train_loss:.5f}  val={val_loss:.5f}")
                else:
                    print(f"[{m+1}/{self.n_estimators}] train_logloss={train_loss:.5f}")

            # Early stopping
            if val_loss is not None and self.early_stopping_rounds > 0:
                if val_loss < best_val_loss - 1e-8:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {m+1}. Best val loss = {best_val_loss:.5f}")
                    break

    def predict_raw(self, X):
        F = np.full(X.shape[0], self.base_score)
        for tree, leaf_w, col_idx in zip(self.trees, self.leaf_weights, self.col_indices):
            leaves = tree.apply(X[:, col_idx])
            F += self.learning_rate * np.array([leaf_w.get(l, 0.0) for l in leaves])
        return F

    def predict_proba(self, X):
        return sigmoid(self.predict_raw(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------
# DATA LOADING + PREPROCESSING
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)

df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df['Married'] = df['Married'].map({'Yes':1,'No':0})
df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0})
df['Property_Area'] = df['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})

df['Dependents'] = df['Dependents'].replace('3+', '3').astype(float)

# Keep your original mapping; change if you want Y=1, N=0
df[TARGET] = df[TARGET].map({'Y':0,'N':1})

for c in num_features:
    df[c] = df[c].fillna(df[c].median())
for c in cat_features:
    df[c] = df[c].astype(str).fillna('Missing')

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

X_df = df[FEATURES]
y = df[TARGET].values

X_temp, X_test_df, y_temp, y_test = train_test_split(
    X_df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Validation split
if VALIDATION_FOR_EARLY_STOP > 0:
    val_frac = VALIDATION_FOR_EARLY_STOP / (1 - TEST_SIZE)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=RANDOM_STATE
    )
else:
    X_train_df, X_val_df, y_train, y_val = X_temp, None, y_temp, None

preprocessor.fit(X_train_df)

X_train = preprocessor.transform(X_train_df).astype(float)
X_val = preprocessor.transform(X_val_df).astype(float) if X_val_df is not None else None
X_test = preprocessor.transform(X_test_df).astype(float)


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

model = RefinedXGBLike(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    lambda_reg=LAMBDA,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    random_state=RANDOM_STATE,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose=True,
    gamma=GAMMA,
    min_child_weight=MIN_CHILD_WEIGHT
)

print("Training refined XGBoost-like model...")
start = time.time()
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
end = time.time()
print(f"Training completed in {end-start:.1f}s")


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)

y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

print("\nTRAIN METRICS")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
try:
    print("ROC AUC:", roc_auc_score(y_train, y_train_proba))
except Exception as e:
    print("ROC AUC: could not compute -", e)

print("\nTEST METRICS")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (test):")
print(classification_report(y_test, y_test_pred))


# ---------------------------------------------------------
# SAVE MODEL + PREPROCESSOR
# ---------------------------------------------------------

full_pipeline = {'preprocessor': preprocessor, 'model': model}
joblib.dump(full_pipeline, MODEL_OUT)

print(f"\nSaved pipeline to: {MODEL_OUT}")
