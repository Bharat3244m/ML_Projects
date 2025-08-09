import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model = load_object('artifacts/model.pkl')
        self.scaler = load_object('artifacts/preprocessor.pkl')

    def predict(self, data):
        try:
            scaled_data = self.scaler.transform(data)
            # Make prediction
            prediction = self.model.predict(scaled_data)
            return prediction
        except Exception as e:
            raise CustomException("Error occurred during prediction", e) # type: ignore

class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int
                 ):
        self.data = {
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "parental_level_of_education": parental_level_of_education,
            "lunch": lunch,
            "test_preparation_course": test_preparation_course,
            "reading_score": reading_score,
            "writing_score": writing_score
        }
        

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame([self.data])
            return df
            logging.info("Data converted to DataFrame successfully")
        except Exception as e:
            raise CustomException("Error converting data to DataFrame", e) # type: ignore
            logging.error(f"Error loading object from {file_path}: {e}")