import pandas as pd
import logging
import warnings

# Suppress FutureWarning for downcasting
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataPreprocessing:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def convert_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts object columns to numeric, timestamp to datetime, and extracts time-based features.
        """
        logging.info("Starting column conversion...")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        logging.info("Column names cleaned of leading/trailing whitespaces.")

        # Convert 'timestamp' column to datetime
        if "timestamp" in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            logging.info("Converted 'timestamp' column to datetime.")

            # Extract useful features from timestamp
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            logging.info("Extracted 'hour', 'dayofweek', 'month', and 'is_weekend' from timestamp.")

        # Now exclude 'timestamp' from object columns to be converted to numeric
        object_cols = df.select_dtypes(include=['object']).columns.drop(['timestamp'], errors='ignore')

        for col in object_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info(f"Converted column '{col}' from object to numeric.")
        df.drop("timestamp", axis=1, inplace=True, errors='ignore')

        return df



    def find_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find and log duplicate rows in the dataframe.
        """
        logging.info("Finding duplicate rows...")
        duplicate_rows = df[df.duplicated()]
        if not duplicate_rows.empty:
            logging.info(f"Found {duplicate_rows.shape[0]} duplicate rows.")
        else:
            logging.info("No duplicate rows found.")
        return df

    def remove_duplicated(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from the dataframe.
        """
        logging.info("Removing duplicate rows...")
        df = df.drop_duplicates()
        logging.info(f"Removed duplicates. New shape: {df.shape}")
        return df

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the dataframe using median.
        """
        logging.info("Filling missing values with median...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                median = df[col].median()
                df.loc[:, col] = df[col].fillna(median)
                logging.info(f"Filled missing values in '{col}' with median: {median}")
        logging.info("Missing value filling completed.")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "data/processed_data.csv") -> None:
        """
        Saves the processed DataFrame to a CSV file.
        """
        logging.info(f"Saving processed data to '{filename}'...")
        df.to_csv(filename, index=False)
        logging.info(f"Data saved successfully to '{filename}'.")

