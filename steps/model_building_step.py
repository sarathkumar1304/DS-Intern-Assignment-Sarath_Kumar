import os
import logging
from typing import Annotated
import mlflow
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model
from zenml.enums import ArtifactType

# Import ModelBuilding class
from src.model_building import RegressionModelBuilder

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

# Define model metadata
model_metadata = Model(
    name="Enegy_Consumption_Prediction_Model",
    version=None,
    license="Apache-2.0",
    description="A machine learning model for energy consumption prediction.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model_metadata)
def model_builder_step(model_name: str, X_train: pd.DataFrame, y_train: pd.Series,tune: bool=False) -> Annotated[
    Pipeline,ArtifactConfig(artifact_type=ArtifactType.MODEL)
]:
    """
    ZenML step to create, preprocess, train, and return a specified model.

    Parameters
    
    model_name : str
        Name of the model to create.
    X_train : pd.DataFrame
        Training data features.
    y_train : pd.Series
        Training data labels/target.

    Returns
    
    Any
        The trained model or pipeline including preprocessing.

    """

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")
    logging.info("Starting model building step...")
    
    if not mlflow.active_run():
        mlflow.start_run()
    
    # Initialize the ModelBuilding class and select model by name
    model_builder = RegressionModelBuilder()
    
    try:
        mlflow.sklearn.autolog()
        model = model_builder.get_model(model_name, X_train, y_train,tune = tune)
        logging.info(f"Model '{model_name}' has been successfully created.")
        # Define the pipeline including the model (assuming no preprocessing here)
        pipeline = Pipeline(steps=[("model", model)])
        # Train the model
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")
        model_builder.save_model(pipeline, model_name)
    except ValueError as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        # end the mlflow run
        mlflow.end_run()
    
    return pipeline

    

    

   
        

    # # Save the model pipeline locally after evaluation
    # model_dir = "models"
    # os.makedirs(model_dir, exist_ok=True)  # Ensure the models directory exists
    # model_path = os.path.join(model_dir, "model.pkl")
    # joblib.dump(pipeline, model_path)  # Save model pipeline as 'model.pkl'
    # logger.info(f"Model saved at {model_path}")


# zenml stack register energy_compustation_stack -a default -o default -d energy_model_deployer -e enegry_tracker --set