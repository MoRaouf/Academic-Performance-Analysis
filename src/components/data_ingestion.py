"""
Contains functionality for getting data from source and save it locally.
"""

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """A class to store data path variables"""
    stores_data_path: str = os.path.join("data", "raw", "stores.csv")
    features_data_path: str = os.path.join("data", "raw", "features.csv")
    sales_data_path: str = os.path.join("data", "raw", "sales.csv")
    raw_data_path: str = os.path.join("data", "raw", "raw.csv")
    prediction_data_path: str = os.path.join("data", "raw", "prediction.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    def __init__(self):
        """Get the paths from DataIngestionConfig()"""
        
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        """Initiate the data ingestion component"""

        logging.info("Initiated the data ingestion component")

        try:
            #read the raw data as dataframe. This can be read from any source (local csv, from database, from API ...etc.)
            stores_df = pd.read_csv(self.ingestion_config.stores_data_path)
            features_df = pd.read_csv(self.ingestion_config.features_data_path)
            sales_df = pd.read_csv(self.ingestion_config.sales_data_path)
            logging.info("Read the raw data as DataFrames")

            #create the /data/raw/ & /data/processed/ directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.prediction_data_path), exist_ok=True)
            
            logging.info("Started merging DataFrames")

            #merge dataframes to create raw dataframe
            raw_df = pd.merge(sales_df, features_df, how="left", on=['Store','Date', 'IsHoliday'])
            raw_df = pd.merge(raw_df, stores_df, how='left', on=['Store'])

            #create prediction dataframe
            features_df_new_year = features_df[features_df.Date > "2012-10-26"]
            dept_df = sales_df.drop(["Date", "Weekly_Sales", "IsHoliday"], axis=1)
            prediction_df = pd.merge(features_df_new_year, dept_df, how="left", on=['Store'])
            prediction_df = pd.merge(prediction_df, stores_df, how='left', on=['Store'])
            
            #save the raw dataframe
            raw_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            prediction_df.to_csv(self.ingestion_config.prediction_data_path, index=False, header=True)

            logging.info("Saved raw & prediction DataFrames")

            #Split the raw data into train & test data
            logging.info("Started splitting data into train & test parts")
            train_data, test_data = train_test_split(raw_df, test_size=0.2, random_state=11)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is completed.")

            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path,
                    self.ingestion_config.prediction_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)
        


# if __name__ == "__main__":

#     data_ingestion = DataIngestion()
#     train_data, test_data, prediction_data = data_ingestion.initiate_data_ingestion()