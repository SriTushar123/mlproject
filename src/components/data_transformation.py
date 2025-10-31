import pandas as pd 
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os, sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scalar",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_Hot_Encoder",OneHotEncoder(drop="first")),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessing_obj=self.get_data_transformer()
            target_feature="math_score"

            #X_train
            input_feature_train_df=train_df.drop(columns=[target_feature],axis=1)
            #X_test
            input_feature_test_df=test_df.drop(columns=[target_feature],axis=1)

            #Y_train
            target_feature_train_df=train_df[target_feature]  
            #Y_test
            target_feature_test_df=test_df[target_feature]  

            ## Starting the Preprocessing 

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            ## Combining test and train datasets with output columns

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)







