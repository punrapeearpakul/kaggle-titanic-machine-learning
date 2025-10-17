from typing import List, Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator
from models.BasePipeline import *

class GeneralModelPipeline(BasePipeline):
    """
    Flexible ML pipeline that allows user to define:
    - Any sklearn-compatible predictor (e.g., RandomForestClassifier)
    - Feature columns to use
    - Custom hyperparameters for the predictor
    """

    def __init__(self, predictor_class: Any, feature_cols: List[str], predictor_hyper_params: Dict = None):
        """
        Parameters
        ----------
        predictor_class : sklearn-like estimator class
            e.g., RandomForestClassifier, LogisticRegression, etc.
        feature_cols : List[str]
            List of feature column names to be used in training and prediction.
        predictor_hyper_params : Dict
            Dictionary of hyperparameters to initialize the predictor.
        """
        super().__init__()
        self.feature_cols = feature_cols
        self.predictor_class = predictor_class
        self.predictor_hyper_params = predictor_hyper_params or {}

    def _transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        No transformation in this pipeline — simply selects feature columns.
        """
        return data[self.feature_cols].copy()

    def _fit_predictor(self, data: pd.DataFrame):
        """
        Instantiate and fit the predictor using provided hyperparameters.
        """
        model: BaseEstimator = self.predictor_class(**self.predictor_hyper_params)
        model.fit(data, self.raw_target)
        return model

    def fit(self, data: pd.DataFrame, target_col: str):
        """
        Fit pipeline:
        1. Save target column
        2. Transform data
        3. Fit model
        """
        self.raw_target = data[target_col]
        return super().fit(data)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using trained model.
        Select only self.feature_cols and return a DataFrame
        with the same index as the input.
        """
        transformed = self._transformed(data)
        preds = self.trained_model.predict(transformed)
        return pd.DataFrame(preds, index=data.index, columns=["prediction"])
