"""
scalers.py — Custom transformer used during model training.

This file MUST be present in the backend package so that joblib can
deserialise the .pkl files. The class was serialised as __main__.NamedRobustScaler
in the training notebook; importing it here before joblib.load() runs
puts it in the correct namespace for unpickling.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class NamedRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=self.columns_,
            index=X.index,
        )
