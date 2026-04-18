# backend/scalers.py
from sklearn.preprocessing import RobustScaler

class NamedRobustScaler(RobustScaler):
    def __init__(self, feature_names=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = feature_names
