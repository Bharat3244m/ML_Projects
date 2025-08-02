from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
import os
import numpy as np   
import pandas as pd  
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from dataclasses import dataclass
from src.utils import save_object
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_data_file_path = os.path.join('artifacts', 'transformed_data.csv')
    class_labels_file_path = os.path.join('artifacts', 'class_labels.csv')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info("Data Transformation method starts")
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Train and Test datasets read as pandas DataFrames")

            # Identify numerical and categorical columns
            numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

            
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()


            target_column = 'math_score'
            
            # Check if target_column exists
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise CustomException(f"{target_column} not found in dataset columns", sys) # type: ignore
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            # 🚨 Recalculate cols AFTER dropping target
            numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            # Define preprocessing for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )
            
            logging.info("Preprocessing pipelines created for train and test data")
            
            
            
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]
            
            logging.info("Data transformation completed successfully")
            
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor object saved successfully")
            
            return (train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)  # type: ignore
        
        
        
        