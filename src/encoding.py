"""Shared model-ready encoding.

Converts the engineered feature DataFrame into a numeric matrix with a
preprocessor fitted ONLY on the training fold, so no test/val information leaks.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pipeline as P

# Categorical columns in our engineered feature set
CATEGORICALS = ["sex", "division", "district", "upzmunc", "ccc", "travel",
                "vax_status", "age_group"]
NUMERICAL_COLS = ["age", "doses_mcv", "doses_rcv", "time_since_mcv",
                  "fever_duration", "rash_duration", "invest_lag", "epiweek",
                  "fever_gte7", "rash_gte3", "ccc_yes",
                  "fever_cough_coryza_conj", "fever_and_rash"]


def make_preprocessor(numeric_cols=None, categorical_cols=None,
                      scale_numeric=True):
    """ColumnTransformer: impute numericals, one-hot categoricals.

    Returns (preprocessor, list-of-feature-columns-kept).
    """
    numeric_cols = numeric_cols or [c for c in NUMERICAL_COLS]
    categorical_cols = categorical_cols or list(CATEGORICALS)

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
    ])
    if scale_numeric:
        num_pipe.steps.append(("scaler", StandardScaler()))

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    return pre, numeric_cols, categorical_cols


def fit_transform(pre, df, cols, target_col="target"):
    X = df[cols].copy()
    return pre.fit_transform(X)


class EncoderWrapper:
    """A reusable encoder that fits once on training and transforms any set."""

    def __init__(self, numeric_cols=None, categorical_cols=None,
                 scale_numeric=True):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.pre = None
        self.scale_numeric = scale_numeric

    def fit(self, df):
        self.pre = make_preprocessor(
            self.numeric_cols, self.categorical_cols,
            scale_numeric=self.scale_numeric)[0]
        X = df[P.get_feature_columns() if not self.numeric_cols else self._cols(df)].copy()
        self.pre.fit(X)
        return self

    def _cols(self, df):
        return self.numeric_cols + self.categorical_cols

    def transform(self, df):
        X = df[self._cols(df)].copy()
        return self.pre.transform(X)

    def get_feature_names(self):
        return self.pre.get_feature_names_out().tolist()


def feature_matrix(df, cols):
    """Raw numeric+onehot without pipeline (for analysis)."""
    pre = make_preprocessor(
        [c for c in cols if c in NUMERICAL_COLS],
        [c for c in cols if c in CATEGORICALS],
        scale_numeric=False)[0]
    return pre.fit_transform(df[cols]), pre.get_feature_names_out().tolist()
