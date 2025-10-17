from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict,List

class BasePipeline(ABC):
    """
    Abstract base class for building machine learning pipelines.
    
    Subclasses must implement:
    - _transformed: to transform or preprocess training data
    - _fit_predictor: to train a model using the transformed data
    """

    def __init__(self,
                 feature_cols: List,
                 predictor_hyper_params: Dict,
                 ):
        # use init attributes
        self.feature_cols: List = feature_cols
        self.predictor_hyper_params: Dict = predictor_hyper_params
        
        # Interal Attributes
        self.transformed_train_data = None
        self.trained_model = None

    @abstractmethod
    def _transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for transforming input data.
        Should return a transformed pandas DataFrame.
        """
        pass

    @abstractmethod
    def _fit_predictor(self, data: pd.DataFrame):
        """
        Abstract method for fitting the model.
        Should return a trained model object.
        """
        pass

    def fit(self, data: pd.DataFrame):
        """
        Main method to train the pipeline.
        1. Transform data using _transformed
        2. Save the result in `transformed_train_data`
        3. Fit the model using _fit_predictor
        4. Save the trained model in `trained_model`
        """
        # Step 1: Transform training data
        self.transformed_train_data = self._transformed(data)

        # Step 2: Train model
        self.trained_model = self._fit_predictor(self.transformed_train_data)

        return self.trained_model

    @abstractmethod
    def predict(self, data: pd.DataFrame):
        """
        Abstract method for making predictions.
        Should take a pandas DataFrame and return predicted results.
        """
        pass
