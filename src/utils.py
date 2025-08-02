import os 
import numpy as np  
import sys
import pandas as pd  
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys) # type: ignore

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for model_name, model in models.items():
            if model_name not in param:
                logging.warning(f"No params found for model: {model_name}, skipping...")
                continue

            logging.info(f"Training model: {model_name} with params: {param[model_name]}")
            
            gs = GridSearchCV(model, param[model_name], cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise CustomException(e, sys) # type: ignore
