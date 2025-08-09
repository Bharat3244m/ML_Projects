#!/usr/bin/env python3
"""
Script to retrain the model with current scikit-learn version
"""
import os
import sys
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

def retrain_model():
    """
    Retrain the model with current scikit-learn version
    """
    try:
        print("Starting model retraining process...")
        
        # Step 1: Data Ingestion
        print("1. Loading and splitting data...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print("✓ Data ingestion completed")
        
        # Step 2: Data Transformation
        print("2. Transforming data...")
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        print("✓ Data transformation completed")
        
        # Step 3: Model Training
        print("3. Training model...")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.train_model(train_array, test_array)
        print(f"✓ Model training completed with R2 score: {r2_score}")
        
        print("\n🎉 Model retraining completed successfully!")
        print(f"Model saved to: artifacts/model.pkl")
        print(f"Preprocessor saved to: artifacts/preprocessor.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model retraining: {str(e)}")
        logging.error(f"Error during model retraining: {str(e)}")
        return False

if __name__ == "__main__":
    success = retrain_model()
    if success:
        print("\n✅ You can now run your Streamlit app!")
    else:
        print("\n❌ Model retraining failed. Please check the logs for details.")