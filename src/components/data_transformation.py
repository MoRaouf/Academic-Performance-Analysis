"""
Contains functionality for transforming data for training.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_pickle_object


@dataclass
class DataTransformationConfig:
    """A class to store preprocessor path"""
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """A class to transform the data for training & prediction"""

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_data_preprocessor(self):
        """Create the preprocessor pipeline of the data columns transformation"""

        try:
            #create or select the numerical & categorical column names
            num_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
            cat_cols = ["Type", "IsHoliday"]

            logging.info(f"Categorical columns to transform: {cat_cols}")
            logging.info(f"Numerical columns to transform: {num_cols}")

            #create the artifacts & preprocessors directories
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_path), exist_ok=True)

            #create numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="mean"))
                ]
            )

            #create categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                ("encoder", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoding", OrdinalEncoder())
                ]
            )

            #create data columns preprocessor
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    
    def initiate_data_transformation(self, train_path, test_path):
        """Initiate the data transformation component"""

        logging.info("Initiated the data transformation component")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path) 

            num_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
            cat_cols = ["Type", "IsHoliday"]
            markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

            #preprocess Date column
            train_df["Date"] = pd.to_datetime(train_df["Date"])
            train_df["Month"] = train_df.Date.dt.month
            train_df["Year"] = train_df.Date.dt.year
            train_df["WeekOfYear"] = train_df.Date.dt.weekofyear
            train_df = train_df.drop("Date", axis=1)

            logging.info(f"Date Column preprocessed")

            #preprocess Markdown columns
            train_df[markdown_cols] = train_df[markdown_cols].interpolate(limit_direction="both")

            #get the preprocessor object
            preprocessor = self.create_data_preprocessor()

            #drop the target column from train & test data
            target_col = "Weekly_Sales"
            y_train = pd.Series(train_df[target_col], name=target_col)
            y_test = pd.Series(test_df[target_col], name=target_col)

            X_train = train_df.drop(target_col, axis=1)
            X_test = train_df.drop(target_col, axis=1)

            #preprocess the data
            preprocessed_X_train = pd.DataFrame(preprocessor.fit_transform(X_train),
                                                columns = num_cols+cat_cols)
            preprocessed_X_test = pd.DataFrame(preprocessor.fit_transform(X_test),
                                               columns = num_cols+cat_cols)
            #remove unncesseary columns
            preprocessed_X_train = pd.concat([X_train.drop(num_cols+cat_cols, axis=1), preprocessed_X_train], axis=1)
            preprocessed_X_test = pd.concat([X_test.drop(num_cols+cat_cols, axis=1), preprocessed_X_test], axis=1)
            
            #scale the data
            scaler = StandardScaler()

            preprocessed_X_train = pd.DataFrame(scaler.fit_transform(preprocessed_X_train), 
                                                columns=preprocessed_X_train.columns)
            preprocessed_X_train = pd.DataFrame(scaler.transform(preprocessed_X_train), 
                                                columns=preprocessed_X_test.columns)
            
            logging.info(f"Preprocessed X_train & X_test")

            #re-add the target column
            final_X_train = pd.concat([preprocessed_X_train, y_train], axis=1)
            final_X_test = pd.concat([preprocessed_X_test, y_test], axis=1)

            save_pickle_object(preprocessor ,self.data_transformation_config.preprocessor_path)

            return (
                final_X_train,
                final_X_test
            )

        except Exception as e:
            raise CustomException(e, sys)