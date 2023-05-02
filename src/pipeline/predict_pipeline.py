"""
A prediction pipeline for the web application.
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from logger import logging
from src.exception import CustomException
from src.utils import load_pickle_object

from sklearn.preprocessing import StandardScaler

@dataclass
class PredictPipelineConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
    model_path = os.path.join("artifacts","model.pkl")

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
            num_features = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
            cat_features = ["Type", "IsHoliday"]

            #load pickle objects
            preprocessor = load_pickle_object(self.predict_pipeline_config.preprocessor_path) 
            model = load_pickle_object(self.predict_pipeline_config.model_path)
            
            #preprocess the data
            preprocessed_X = pd.DataFrame(preprocessor.transform(X),
                                          columns = num_features+cat_features)
            #remove unncesseary columns
            preprocessed_X = pd.concat([X.drop(num_features+cat_features, axis=1), preprocessed_X], axis=1)
            
            #scale the data
            scaler = StandardScaler()
            preprocessed_X = pd.DataFrame(scaler.fit_transform(preprocessed_X),
                                          columns=preprocessed_X.columns)
            
            #load the model
            model = load_pickle_object(self.predict_pipeline_config.model_path)

            #get predictions
            preds = model.predict(preprocessed_X)

            logging.info("Finished prediction pipeline")

            return preds


        except Exception as e:
            raise CustomException(e, sys)