import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor 
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def train_model(self, train_array, test_array):
        try:
            logging.info("splitting data into train and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info("Started Model Training...")
            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'XGBRegressor': XGBRegressor(use_label_encoder=False, eval_metric='rmse'),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'DecisionTreeRegressor': DecisionTreeRegressor()
            }
            
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor":{
                    'n_neighbors':[3,5,7,9,11],
                    'weights':['uniform','distance'],
                    'algorithm':['auto','ball_tree','kd_tree','brute']
                },
                "LogisticRegression":{
                    'penalty':['l2','l1','elasticnet'],
                    'C':[0.1,0.5,1,5,10],
                    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                }
                
            }


            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train
                                                 , X_test=X_test, y_test=y_test, models=models, param=params)
            
            # getting best model score from the report
            best_model_score = max(sorted(model_report.values()))
            
            # getting best model from the report
            best_model_name = [key for key, val in model_report.items() if val == best_model_score]
            
            best_model = models[best_model_name[0]]
            
            if best_model_score < 0.75:
                raise CustomException("No best model found with sufficient score", sys) # type: ignore
            
            logging.info(f"Best model found: {best_model_name[0]} with score: {best_model_score}")
            
            save_object(file_path=self.model_trainer_config.model_dir, obj=best_model)
            
            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
    