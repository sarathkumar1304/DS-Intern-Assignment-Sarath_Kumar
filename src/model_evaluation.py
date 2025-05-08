import logging
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    def __init__(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initializes the ModelEvaluator with model, test data, and experiment details.

        Parameters:
        model : BaseEstimator
            The trained model to evaluate.
        X_test : pd.DataFrame
            The test features.
        y_test : pd.Series
            The true labels for the test set.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
   

        logging.info("ModelEvaluator initialized.")

    def evaluate_model(self) -> dict:
        """
        Evaluates the model and logs regression metrics.

        Returns:
        -------
        dict
            A dictionary containing MSE, MAE, RMSE, and R².
        """
        logging.info("Starting model evaluation...")

        # Make predictions on the test set
        try:
            y_pred = self.model.predict(self.X_test)
            logging.info("Model prediction successful.")
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

        # Calculate and log evaluation metrics for regression
        try:
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            logging.info("Metrics calculated successfully.")
        except Exception as e:
            logging.error(f"Error in calculating metrics: {e}")
            raise

        # Store metrics in a dictionary
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

        # Log each metric
        logging.info("Logging evaluation metrics.")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name.upper()}: {metric_value:.4f}")

        return metrics
