from zenml import step
import pandas as pd
import logging
from src.data_preprocessing import DataPreprocessing  # Ensure this import path is correct

# # Configure logging
# logging.basicConfig(
#     filename="pipeline.log", 
#     level=logging.INFO, 
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

@step(enable_cache=False)
def data_preprocessing_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML step to preprocess input data using the DataPreprocessing class.

    Args:
        df (pd.DataFrame): Raw input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    try:
        logging.info("Running data_preprocessing_step...")

        # Instantiate the DataPreprocessing class
        preprocessor = DataPreprocessing()
        
        # Call the preprocessing methods from DataPreprocessing class
        processed_df = preprocessor.convert_column(df)  # Converts columns as per preprocessing rules
        

        processed_df = preprocessor.find_duplicate_rows(processed_df)  # Assuming you implement this function
        processed_df = preprocessor.remove_duplicated(processed_df)    # Assuming you implement this function
        processed_df = preprocessor.fill_missing_values(processed_df)  # Assuming you implement this function
        preprocessor.save_to_csv(df, "data/final_processed_data.csv")

        logging.info("Data preprocessing step completed successfully.")
        return processed_df

    except Exception as e:
        logging.exception("data_preprocessing_step failed.")
        raise RuntimeError("Data preprocessing step failed.") from e
