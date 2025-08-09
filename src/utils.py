import os 
import numpy as np  
import sys
import pandas as pd  
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill

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


def load_object(file_path):
    """
    Load an object from a file using pickle with comprehensive version compatibility handling.
    """
    # List of methods to try for loading the object
    load_methods = [
        ('dill', lambda f: dill.load(f)),
        ('pickle', lambda f: pickle.load(f))
    ]
    
    for method_name, load_func in load_methods:
        try:
            with open(file_path, 'rb') as file:
                obj = load_func(file)
            logging.info(f"Object loaded successfully using {method_name} from {file_path}")
            return obj
        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            # Handle scikit-learn version compatibility issues
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['_loss', 'sklearn', 'scikit', 'version', 'compatibility']):
                logging.warning(f"Detected version compatibility issue with {method_name}. Trying next method...")
                continue
            else:
                logging.warning(f"Error with {method_name}: {e}. Trying next method...")
                continue
        except Exception as e:
            logging.warning(f"Unexpected error with {method_name}: {e}. Trying next method...")
            continue
    
    # If all methods fail, provide a detailed error message
    error_msg = f"""
    Failed to load object from {file_path}. This is likely due to:
    1. Scikit-learn version incompatibility between training and inference environments
    2. Missing dependencies
    3. Corrupted model file
    
    Solutions:
    1. Retrain the model with the current scikit-learn version
    2. Install the same scikit-learn version used during training
    3. Check if all required dependencies are installed
    
    Current error: {str(e)}
    """
    logging.error(error_msg)
    raise CustomException(error_msg, sys) # type: ignore