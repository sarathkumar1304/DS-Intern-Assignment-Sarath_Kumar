from src.data_ingestion import DataIngestion
from zenml import step
import pandas as pd
import logging



@step
def data_ingestion_step(path: str) -> pd.DataFrame:
    """
    ZenML step to ingest data from a CSV file using the DataIngestion class.

    Args:
        path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        logging.info(f"Running data_ingestion_step for file: {path}")
        ingestion = DataIngestion()
        df = ingestion.ingest_data(path)
        logging.info(f"Data ingestion step completed. Shape: {df.shape}")
        return df

    except Exception as e:
        logging.exception(f"Error in data_ingestion_step: {str(e)}")
        raise RuntimeError("Data ingestion failed.") from e
