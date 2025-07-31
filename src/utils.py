import os 
import numpy as np  
import sys
import pandas as pd  
from src.logger import logging
from src.exception import CustomException
import pickle

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