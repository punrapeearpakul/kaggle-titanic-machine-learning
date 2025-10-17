from sklearn.ensemble import RandomForestClassifier
from models.GeneralModelPipeline import *

class RandomForestPipelineV1(GeneralModelPipeline):
    """
    Specialized pipeline subclass using RandomForestClassifier as predictor.
    - Feature columns are all columns.
    - Predictor hyperparameters are fixed (cannot be overridden).
    """

    def __init__(self):

        feature_cols = [
            'Pclass', 'Sex', 'Age', 'SibSp',
            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
        ]


        fixed_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': -1
        }

        super().__init__(
            predictor_class=RandomForestClassifier,
            feature_cols=feature_cols,
            predictor_hyper_params=fixed_params
        )
