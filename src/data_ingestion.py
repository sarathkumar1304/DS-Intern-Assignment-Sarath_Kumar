import logging
import pandas as pd
from pandas import DataFrame

# Configure logging once at the top-level (only once per app/module)
logging.basicConfig(
    filename="data_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataIngestion:
    """
    Handles data ingestion from CSV files using pandas.
    """

    def ingest_data(self, path: str) -> DataFrame:
        """
        Reads a CSV file from the given path and returns a DataFrame.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame from the CSV.

        Raises:
            FileNotFoundError: If the CSV file is not found.
            pd.errors.ParserError: If the CSV file is corrupted.
            Exception: For any other unexpected errors.
        """
        try:
            logging.info(f"Starting to read data from {path}")
            df = pd.read_csv(path)
            logging.info(f"Successfully read {len(df)} rows and {len(df.columns)} columns from {path}")
            return df

        except FileNotFoundError:
            logging.error(f"File not found at path: {path}")
            raise

        except pd.errors.ParserError:
            logging.error(f"Parsing error while reading the CSV at path: {path}")
            raise

        except Exception as e:
            logging.exception(f"Unexpected error during data ingestion from {path}: {str(e)}")
            raise


# if __name__ == "__main__":
#     ingestion = DataIngestion()
#     df = ingestion.ingest_data("data/data.csv")
#     print(df.head())
