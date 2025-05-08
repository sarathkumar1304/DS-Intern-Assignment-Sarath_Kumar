import logging
import numpy as np
from zenml.steps import step
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator
from src.model_evaluation import ModelEvaluator  # Ensure this import path is correct
import pandas as pd
@step
def model_evaluation_step(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    A ZenML step to evaluate the trained model.

    Parameters:
    model: BaseEstimator
        The trained model to evaluate.
    X_test: pd.DataFrame
        The test features.
    y_test: pd.Series
        The true labels for the test set.
    
    Returns:
    -------
    dict
        A dictionary containing evaluation metrics.
    """
    evaluator = ModelEvaluator(model, X_test, y_test)
    return evaluator.evaluate_model()
