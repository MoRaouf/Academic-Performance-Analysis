"""
A training pipeline for training & evluating models.
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_pickle_object, evaluate_model, read_yaml

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV



@dataclass
class TrainPipelineConfig:
    """A class to store model path"""
    trained_model_path=os.path.join("artifacts", "model.pkl")
    params_path = os.path.join("artifiacts", "params.yaml")


class TrainPipeline:
    """A class to train the data"""
    def __init__(self):
        self.train_pipeline_config = TrainPipelineConfig()

    def initiate_train_pipeline(self, train_data, test_data):
        """Initiate the training pipeline"""
        try:
            logging.info("Started training pipeline")
            #Create X_train, y_train
            X_train = train_data.drop("Weekly_Sales", axis=1)
            y_train = train_data["Weekly_Sales"]

            #Create X_test, y_test
            X_test = test_data.drop("Weekly_Sales", axis=1)
            y_test = test_data["Weekly_Sales"]

            #create XGBRegressor model
            xgb_model = XGBRegressor()

            #read the params yaml file
            params = read_yaml(self.model_trainer_config.params_path)

            #create GridSearchCV
            grid_search = GridSearchCV(xgb_model, param_grid=params, cv=3, n_jobs=-1)
            #fit the model on all features
            grid_search.fit(X_train, y_train)

            #get the best model from the GridSearchCV
            gs_best_model = grid_search.best_estimator_

            #get predictions for train & test data
            pred_train = gs_best_model.predict(X_train)
            pred_test = gs_best_model.predict(X_test)

            #evaluate Train and Test dataset
            rmse_train, r2_train = evaluate_model(y_train, pred_train)
            rmse_test, r2_test = evaluate_model(y_test, pred_test)

            #create a series of scores
            model_scores = pd.Series([rmse_train, rmse_test, r2_train, r2_test], 
                                        name="XGBoost Regressor", 
                                        index=["RMSE-train", "RMSE-test", "R2-train", "R2-test"]).to_frame().T


            if model_scores["R2-test"]<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset")


            save_pickle_object(
                gs_best_model,
                self.model_trainer_config.trained_model_path
            )

            return model_scores

        except Exception as e:
            raise CustomException(e, sys)
