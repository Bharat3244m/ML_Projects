import streamlit as st
import pandas as pd
import os
import pickle
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Page title
st.title("🚀 Student Performance Prediction Model")

gender = st.selectbox('Gender', ['Male', 'Female'], key='gender')
race_ethnicity=st.selectbox('Race_Ethnicity',options=['Group A', 'Group B', 'Group C', 'Group D', 'Group E'], key ='race_ethnicity')
parental_level_of_education=st.selectbox('parental_level_of_education',options=['Some high school', 'High school', 'Some college', 'Associate degree', 'Bachelor’s degree', 'Master’s degree'], key='parental_level_of_education')
lunch=st.selectbox('lunch',options=['Standard', 'Free/reduced'], key='lunch')
test_preparation_course=st.selectbox('test_preparation_course',options=['Completed', 'None'], key='test_preparation_course')
reading_score= st.number_input('Reading Score', min_value=0, max_value=100, step=1, key='reading_score')
writing_score= st.number_input('Writing Score', min_value=0, max_value=100, step=1, key='writing_score')

if st.button('Predict'):
    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        predict_pipeline = PredictPipeline()
        pred_df = data.get_data_as_dataframe()  
        result = predict_pipeline.predict(pred_df)
        st.success(f"Predicted Score: {result[0]}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.error(f"Error during prediction: {str(e)}")

