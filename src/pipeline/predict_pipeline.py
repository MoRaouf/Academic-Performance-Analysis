"""
A prediction pipeline for the web application.
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import load_pickle_object

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

@dataclass
class PredictPipelineConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
    model_path = os.path.join("models","model.pkl")

class PredictPipeline:
    
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()
    
    def predict(self, predict_data):
        """Get the predictions of the passed data"""

        logging.info("Initiated prediction pipeline")
        try:
            #preprocess Date column
            X = predict_data.copy()

            X["Date"] = pd.to_datetime(X["Date"])
            X["Month"] = X.Date.dt.month
            X["Year"] = X.Date.dt.year
            X["WeekOfYear"] = X.Date.dt.weekofyear
            X = X.drop("Date", axis=1)

            #preprocess Markdown columns
            markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
            X[markdown_cols] = X[markdown_cols].interpolate(limit_direction="both")

            #data preprocessing
            num_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
            cat_cols = ["Type", "IsHoliday"]

            #load pickle objects
            # preprocessor = load_pickle_object(self.predict_pipeline_config.preprocessor_path) 
            # model = load_pickle_object(self.predict_pipeline_config.model_path)
            
            #preprocess the data
            # preprocessed_X = pd.DataFrame(preprocessor.transform(X),
            #                               columns = num_features+cat_features)
            #remove unncesseary columns
            # preprocessed_X = pd.concat([X.drop(num_features+cat_features, axis=1), preprocessed_X], axis=1)
            
            #Impute num_cols with the mean of their respective month
            imputer = SimpleImputer(strategy='mean')

            for month in X.Month.unique():
                X.loc[X.Month == month, num_cols]= imputer.fit_transform(X.loc[X.Month == month, num_cols])

            #encode cat_cols in X_train & X_test
            encoder = OrdinalEncoder()
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

            #scale the data
            # scaler = StandardScaler()
            # preprocessed_X = pd.DataFrame(scaler.fit_transform(X),
            #                               columns=X.columns)
            
            #load the model
            model_pkl = load_pickle_object(self.predict_pipeline_config.model_path)
            model = model_pkl["model_object"]

            #get predictions
            preds = model.predict(X)

            logging.info("Finished prediction pipeline")

            return preds


        except Exception as e:
            raise CustomException(e, sys)