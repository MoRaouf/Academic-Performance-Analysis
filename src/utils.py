import os
import sys
import pickle
import yaml

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from logger import logging


def save_pickle_object(obj, file_path):
    """Serialize Pyhton object as pickle object"""
    try:
        #create directory if not yet existing
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logging.info(f"Saving pickle object to: {file_path}")

        #save the object in the file
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_pickle_object(file_path):
    """Load pickle object into python pbject"""

    try:
        #open the file where the serialized object is saved & return it as python object
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys)
    

def read_yaml(yaml_file):
    """Read yaml object"""
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)
    

def set_seeds(seed:int) -> None:
    """Set different seed"""
    np.random.seed(seed)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Evaluate regression models & return R2 score of prediction of the best model"""
    try:
        report = {}
        highest_test_score = 0
        best_model = ""

        for i in range(len(list(models))):
            #get the model & its parameters
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            #apply GridSearchCV 
            grid_search = GridSearchCV(model, para, cv=3)
            grid_search.fit(X_train,y_train)

            #get the best parameters of the GridSearchCV & set them to the model  
            # model.set_params(**grid_search.best_params_)

            #get the best model from the GridSearchCV
            grid_search_best_model = grid_search.best_estimator_

            #get predictions for train & test data
            train_pred = grid_search_best_model.predict(X_train)
            test_pred = grid_search_best_model.predict(X_test)

            #get the scores for train & test predictions
            train_model_score = r2_score(y_train, train_pred)
            test_model_score = r2_score(y_test, test_pred)

            #append the model & its score to the report
            report[list(models.keys())[i]] = test_model_score

            if test_model_score >= highest_test_score:
                highest_test_score = test_model_score
                best_model = grid_search_best_model

        return report, best_model
    

    except Exception as e:
        raise CustomException(e, sys)
