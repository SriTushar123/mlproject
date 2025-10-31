import pandas as pd 
import numpy as np 
import os
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.utils import evaluate_models,save_object
from src.exception import CustomException
from sklearn.metrics import r2_score
import sys

@dataclass
class ModelTrainerConfig:
    model_trained_obj_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            X_train,X_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False,allow_writing_files=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            report=evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)


            sorted_report = dict(sorted(report.items(),key=lambda x:x[1],reverse=True))
            best_model_name,best_model_score=list(sorted_report.items())[0]

            model_obj=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No Best Model found")

            save_object(
                file_path=self.model_trainer_config.model_trained_obj_file_path,
                obj=model_obj
            )
            model_obj.fit(X_train,y_train)
            predict_val=model_obj.predict(X_test)
            score=r2_score(y_test,predict_val)

            return score
        
        except Exception as e:
            raise CustomException(e,sys)





