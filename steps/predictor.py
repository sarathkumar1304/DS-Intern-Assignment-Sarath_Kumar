import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step
def predictor(service: MLFlowDeploymentService, input_data: str) -> np.ndarray:
    """
    Makes predictions using a deployed MLflow model service that includes preprocessing (scaling).

    Args:
        service (MLFlowDeploymentService): MLflow deployment service used for prediction.
        input_data (str): A JSON string containing the input data.

    Returns:
        np.ndarray: Predicted outputs.
    """
    service.start(timeout=60)

    # Load and parse input JSON
    try:
        parsed_input = json.loads(input_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    parsed_input.pop("Columns", None)
    parsed_input.pop("index", None)

    # Define expected feature columns
    expected_columns = [
        "equipment_energy_consumption", "lighting_energy",
        "zone1_temperature", "zone1_humidity", "zone2_temperature", "zone2_humidity",
        "zone3_temperature", "zone3_humidity", "zone4_temperature", "zone4_humidity",
        "zone5_temperature", "zone5_humidity", "zone6_temperature", "zone6_humidity",
        "zone7_temperature", "zone7_humidity", "zone8_temperature", "zone8_humidity",
        "zone9_temperature", "zone9_humidity", "outdoor_temperature", "atmospheric_pressure",
        "outdoor_humidity", "wind_speed", "visibility_index", "dew_point",
        "random_variable1", "random_variable2", "hour", "dayofweek", "month", "is_weekend"
    ]

    # Convert input to DataFrame
    try:
        df = pd.DataFrame(parsed_input["data"], columns=expected_columns)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error constructing DataFrame: {e}")

    # Convert DataFrame to dict format accepted by MLflow
    data_dict = {"instances": df.to_dict(orient="records")}

    # Predict using the deployed MLflow model
    try:
        prediction = service.predict(data_dict)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    return np.array(prediction)
